# csv_to_json_converter.py
"""
CSV到JSON转换工具
将训练数据CSV文件转换为认知链JSON格式
"""

import pandas as pd
import json
import os
from pathlib import Path
import numpy as np
# 添加rapidfuzz库用于模糊匹配
try:
    from rapidfuzz import fuzz, process
except ImportError:
    print("警告: 未安装rapidfuzz库，将自动安装...")
    import sys
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rapidfuzz'])
    from rapidfuzz import fuzz, process

# 获取项目根目录的绝对路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATA_DIR = os.path.join(ROOT_DIR, 'python', 'data')

def analyze_data_gaps(thought_df, emotion_df, action_df, verbose=True):
    """分析数据匹配情况，找出缺失原因"""
    
    # 创建情境-思维的组合键
    thought_df['key'] = thought_df['situation'] + '|' + thought_df['thought']
    emotion_df['key'] = emotion_df['situation'] + '|' + emotion_df['thought']
    action_df['key'] = action_df['situation'] + '|' + action_df['thought']
    
    # 统计各CSV的键集合
    thought_keys = set(thought_df['key'])
    emotion_keys = set(emotion_df['key'])
    action_keys = set(action_df['key'])
    
    # 分析覆盖率
    emotion_coverage = len(thought_keys.intersection(emotion_keys)) / len(thought_keys) * 100
    action_coverage = len(thought_keys.intersection(action_keys)) / len(thought_keys) * 100
    
    if verbose:
        print("\n数据覆盖分析:")
        print(f"思维数据总量: {len(thought_keys)}条")
        print(f"情绪数据覆盖率: {emotion_coverage:.2f}%")
        print(f"行为数据覆盖率: {action_coverage:.2f}%")
        
        # 查看未匹配的样例
        missing_emotion_keys = thought_keys - emotion_keys
        missing_action_keys = thought_keys - action_keys
        
        if missing_emotion_keys:
            print("\n缺失情绪的样例(前3条):")
            sample_count = 0
            for key in list(missing_emotion_keys)[:10]:  # 尝试前10条，找到3条可以显示的
                try:
                    row = thought_df[thought_df['key'] == key].iloc[0]
                    print(f"  情境: {row['situation']}")
                    print(f"  思维: {row['thought']}")
                    sample_count += 1
                    if sample_count >= 3:
                        break
                except:
                    continue
                
        if missing_action_keys:
            print("\n缺失行为的样例(前3条):")
            sample_count = 0
            for key in list(missing_action_keys)[:10]:  # 尝试前10条，找到3条可以显示的
                try:
                    row = thought_df[thought_df['key'] == key].iloc[0]
                    print(f"  情境: {row['situation']}")
                    print(f"  思维: {row['thought']}")
                    sample_count += 1
                    if sample_count >= 3:
                        break
                except:
                    continue
    
    # 检查是否有模糊匹配的可能性
    # 随机抽取几个未匹配的思维记录，与情绪/行为数据进行相似度比较
    if verbose and (missing_emotion_keys or missing_action_keys):
        print("\n检查模糊匹配可能性:")
        
        # 从缺失数据中随机抽取一条
        if missing_emotion_keys:
            sample_key = list(missing_emotion_keys)[0]  
            sample_row = thought_df[thought_df['key'] == sample_key].iloc[0]
            sample_sit = sample_row['situation']
            sample_thought = sample_row['thought']
            
            # 查找相似的情绪数据
            similar_emotions = emotion_df[
                (emotion_df['situation'].str.contains(sample_sit[:20], case=False, na=False)) | 
                (emotion_df['thought'].str.contains(sample_thought[:20], case=False, na=False))
            ]
            
            if len(similar_emotions) > 0:
                print("找到可能的情绪数据匹配项:")
                print(f"  未匹配思维: {sample_sit} | {sample_thought}")
                print(f"  可能匹配项: {similar_emotions.iloc[0]['situation']} | {similar_emotions.iloc[0]['thought']}")
                print("  这表明使用模糊匹配可能会提高覆盖率")
    
    return {
        'emotion_coverage': emotion_coverage,
        'action_coverage': action_coverage,
        'missing_emotion_count': len(thought_keys - emotion_keys),
        'missing_action_count': len(thought_keys - action_keys),
        'missing_emotion_keys': thought_keys - emotion_keys,
        'missing_action_keys': thought_keys - action_keys
    }

def fuzzy_merge(df_base, df_to_merge, on_columns, value_columns, similarity_threshold=85, verbose=False):
    """
    使用模糊匹配的方式合并两个DataFrame
    
    参数:
        df_base: 基础DataFrame
        df_to_merge: 要合并的DataFrame
        on_columns: 用于匹配的列名列表
        value_columns: 要从df_to_merge中获取的值的列名列表
        similarity_threshold: 相似度阈值(0-100)，默认为85
        verbose: 是否输出详细信息
    
    返回:
        合并后的DataFrame
    """
    # 复制DataFrame以避免修改原始数据
    df_base = df_base.copy()
    df_to_merge = df_to_merge.copy()
    
    if verbose:
        print(f"正在使用模糊匹配合并数据...")
        print(f"基础数据量: {len(df_base)}条")
        print(f"需要合并数据量: {len(df_to_merge)}条")
        print(f"相似度阈值: {similarity_threshold}%")
    
    # 对匹配列进行预处理：转小写、去除标点和多余空格
    for col in on_columns:
        if df_base[col].dtype == 'object':
            df_base[col + '_clean'] = df_base[col].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
        if df_to_merge[col].dtype == 'object':
            df_to_merge[col + '_clean'] = df_to_merge[col].str.lower().str.replace(r'[^\w\s]', '', regex=True).str.strip()
    
    # 创建匹配键
    clean_cols = [col + '_clean' for col in on_columns]
    df_base['merge_key'] = df_base[clean_cols].apply(lambda x: ' '.join([str(i) for i in x]), axis=1)
    df_to_merge['merge_key'] = df_to_merge[clean_cols].apply(lambda x: ' '.join([str(i) for i in x]), axis=1)
    
    # 为结果DataFrame准备列，确保使用正确的数据类型
    matched_count = 0
    for col in value_columns:
        # 确保使用与源列相同的数据类型
        if col in df_to_merge.columns:
            # 创建具有正确数据类型的列
            df_base[col] = pd.Series(dtype=df_to_merge[col].dtype)
    
    # 使用rapidfuzz的process.extractOne方法进行模糊匹配
    merge_keys = df_to_merge['merge_key'].tolist()
    merge_key_to_idx = {key: idx for idx, key in enumerate(merge_keys)}

    # 对每一行基础数据寻找最佳匹配
    for i, row in df_base.iterrows():
        query = row['merge_key']
        
        # 使用process.extractOne方法进行模糊匹配
        match_result = process.extractOne(
            query,
            merge_keys,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=similarity_threshold
        )
        
        # 如果找到匹配项，填充值
        if match_result:
            matched_key, score, _ = match_result
            merge_idx = merge_key_to_idx[matched_key]
            matched_count += 1
            
            # 填充值
            for col in value_columns:
                try:
                    df_base.loc[i, col] = df_to_merge.loc[merge_idx, col]
                except Exception as e:
                    if verbose:
                        print(f"\n警告: 在赋值列 '{col}' 时发生异常: {e}")
                        print(f"  - 源值: {df_to_merge.loc[merge_idx, col]} (类型: {type(df_to_merge.loc[merge_idx, col]).__name__})")
                        print(f"  - 目标列类型: {df_base[col].dtype}")
    
    if verbose:
        print(f"模糊匹配结果: 成功匹配 {matched_count} 条数据")  
        print(f"匹配率: {matched_count/len(df_base)*100:.2f}%")
    
    # 清理临时列
    for col in on_columns:
        if col + '_clean' in df_base.columns:
            df_base.drop(col + '_clean', axis=1, inplace=True)
    if 'merge_key' in df_base.columns:
        df_base.drop('merge_key', axis=1, inplace=True)
    
    return df_base

def convert_csv_to_json(
    csv_dir=DATA_DIR, 
    output_json=None,
    verbose=True
):
    """
    将CSV文件转换为认知链JSON数据
    
    Args:
        csv_dir: CSV文件目录
        output_json: 输出JSON文件路径，如果为None则默认保存到当前目录
        verbose: 是否打印详细信息
    
    Returns:
        bool: 操作是否成功
    """
    try:
        # 如果未指定输出路径，默认保存到脚本所在目录
        if output_json is None:
            output_json = os.path.join(os.path.dirname(__file__), 'cognitive_chains.json')
        
        if verbose:
            print(f"\n=== CSV到JSON转换工具 ====")
            print(f"数据目录: {os.path.abspath(csv_dir)}")
            print(f"输出文件: {os.path.abspath(output_json)}")
        
        # 检查目录是否存在
        if not os.path.exists(csv_dir):
            print(f"\n错误: 目录{csv_dir}不存在:\n{os.path.abspath(csv_dir)}")
            # 打印当前目录结构以帮助用户定位
            current_dir = os.path.dirname(__file__)
            print(f"\n当前脚本所在目录: {current_dir}")
            print("\n目录结构:")
            
            # 打印相关目录结构
            print(f"  {os.path.basename(ROOT_DIR)}/")
            print(f"    python/")
            if os.path.exists(os.path.join(ROOT_DIR, 'python')):
                subdirs = os.listdir(os.path.join(ROOT_DIR, 'python'))
                for subdir in subdirs:
                    if os.path.isdir(os.path.join(ROOT_DIR, 'python', subdir)):
                        print(f"      {subdir}/")
            return False
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            if verbose:
                print(f"创建输出目录: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # 定义CSV文件路径
        thought_file = os.path.join(csv_dir, 'train_thought.csv')
        emotion_file = os.path.join(csv_dir, 'train_emotion.csv')
        action_file = os.path.join(csv_dir, 'train_action.csv')
        
        # 检查文件是否存在
        missing_files = []
        for file_path in [thought_file, emotion_file, action_file]:
            if not os.path.exists(file_path):
                missing_files.append(f"{os.path.basename(file_path)} (路径: {file_path})")
        
        if missing_files:
            print(f"\n错误: 以下文件不存在:\n{chr(10).join(missing_files)}")
            return False
        
        # 读取需要的CSV文件，思维CSV已经包含了线索数据
        df_thought = pd.read_csv(thought_file)  # 包含: situation, clue, thought, polarity
        df_emotion = pd.read_csv(emotion_file)  # 包含: situation, thought, emotion, polarity
        df_action = pd.read_csv(action_file)   # 包含: situation, thought, action, polarity
        
        # 执行数据缺失分析 (阶段一)
        if verbose:
            print("\n执行数据缺失分析...")
        gap_analysis = analyze_data_gaps(df_thought, df_emotion, df_action, verbose)
        
        # 数据清理 - 去除字段中的前后空格和多余引号
        if verbose:
            print("\n开始数据清理...")
        
        # 清理思维数据
        for col in df_thought.columns:
            if df_thought[col].dtype == 'object':
                df_thought[col] = df_thought[col].str.strip()
        
        # 清理情绪数据
        for col in df_emotion.columns:
            if df_emotion[col].dtype == 'object':
                df_emotion[col] = df_emotion[col].str.strip()
                # 特别处理情绪字段，移除引号
                if col == 'emotion':
                    df_emotion[col] = df_emotion[col].str.replace('"', '')
        
        # 清理行为数据
        for col in df_action.columns:
            if df_action[col].dtype == 'object':
                df_action[col] = df_action[col].str.strip()
        
        # 检查列名情况
        if verbose:
            print(f"\n各文件列名:")
            print(f"- 思维CSV列名: {', '.join(df_thought.columns)}")
            print(f"- 情绪CSV列名: {', '.join(df_emotion.columns)}")
            print(f"- 行为CSV列名: {', '.join(df_action.columns)}")
            print(f"\n数据样例:")
            print(f"- 思维数据示例:\n{df_thought.head(2)}")
            print(f"\n- 情绪数据示例:\n{df_emotion.head(2)}")
            print(f"\n- 行为数据示例:\n{df_action.head(2)}")
            print(f"\n数据量:")
            print(f"- 思维数据量: {len(df_thought)}条")
            print(f"- 情绪数据量: {len(df_emotion)}条")
            print(f"- 行为数据量: {len(df_action)}条")
        
        # 使用思维CSV作为基础，因为它已经包含了线索数据
        df_combined = df_thought.copy()
        
        # 检查合并前的数据量
        if verbose:
            print(f"\n合并前思维数据量: {len(df_combined)}条")
            
        # 改进合并策略，确保使用干净的键进行匹配
        if verbose:
            print("\n执行数据合并...")
        
        # 使用模糊匹配合并情绪数据
        df_combined = fuzzy_merge(df_combined, df_emotion, ['situation', 'thought'], ['emotion'], similarity_threshold=85, verbose=verbose)
        
        # 使用模糊匹配合并行为数据
        df_combined = fuzzy_merge(df_combined, df_action, ['situation', 'thought'], ['action'], similarity_threshold=85, verbose=verbose)
        
        # 检查合并后的数据清洁度
        unknown_emotion_count = df_combined['emotion'].isna().sum()
        unknown_action_count = df_combined['action'].isna().sum()
        
        if verbose:
            print("\n合并后的数据清洁度:")
            print(f"总数据量: {len(df_combined)}条")
            print(f"未匹配情绪数量: {unknown_emotion_count}条 (未匹配率: {unknown_emotion_count/len(df_combined)*100:.2f}%)")
            print(f"未匹配行为数量: {unknown_action_count}条 (未匹配率: {unknown_action_count/len(df_combined)*100:.2f}%)")
        
        # 处理缺失值（情绪和行为）- 避免使用 inplace=True
        df_combined['emotion'] = df_combined['emotion'].fillna('Unknown')
        df_combined['action'] = df_combined['action'].fillna('Unknown')
        
        # 设定情绪极性
        emotion_polarity = {
            'happy': 1.0, 'joyful': 1.0, 'excited': 0.8, 'content': 0.6,
            'angry': -0.8, 'frustrated': -0.7, 'annoyed': -0.6,
            'sad': -0.7, 'disappointed': -0.6, 'upset': -0.5,
            'afraid': -0.8, 'anxious': -0.7, 'nervous': -0.6, 'fearful': -0.8,
            'disgusted': -0.7, 'surprised': 0.1, 'neutral': 0.0,
            'unknown': 0.0
        }
        
        # 为每条记录生成一个ID
        df_combined['chain_id'] = [f"chain_{i}" for i in range(len(df_combined))]
        
        # 转换为字典并保存为JSON
        if verbose:
            print(f"\n转换为JSON并保存到 {output_json}...")
            print(f"- 合并后的数据量: {len(df_combined)}条")
        
        cognitive_chains = {}
        for i, row in df_combined.iterrows():
            # 使用情绪作为链ID的一部分，确保情绪值是干净的
            clean_emotion = row['emotion'].strip()
            chain_id = f"chain_{i}_{clean_emotion}"
            
            # 从情绪极性字典中获取极性值
            polarity = row['polarity'] if 'polarity' in row else 0.0
            
            # 如果情绪在极性字典中，使用对应的极性值
            emotion_lower = clean_emotion.lower()
            for emotion_type, pol_value in emotion_polarity.items():
                if emotion_type in emotion_lower:
                    polarity = pol_value
                    break
                    
            cognitive_chains[chain_id] = {
                'situation': row['situation'],
                'clue': row['clue'],
                'thought': row['thought'],
                'emotion': clean_emotion,  # 使用清理后的情绪值
                'action': row['action'],
                'polarity': float(polarity),
                'activation_level': 0.5  # 默认激活水平
            }
        
        # 保存JSON
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(cognitive_chains, f, ensure_ascii=False, indent=4)
        
        if verbose:
            print(f"\n完成! 已创建 {len(cognitive_chains)} 条认知链记录")
            print(f"文件已保存到: {os.path.abspath(output_json)}")
            
            # 打印生成的JSON预览
            print("\n生成的JSON预览 (前3条记录):")
            sample_keys = list(cognitive_chains.keys())[:3]
            for key in sample_keys:
                print(f"\n{key}:")
                for field, value in cognitive_chains[key].items():
                    print(f"  {field}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    # 设置数据目录和输出文件
    data_dir = DATA_DIR
    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, 'cognitive_chains.json')
    
    # 运行转换
    success = convert_csv_to_json(data_dir, output_file)
    
    if success:
        # 预览生成的JSON内容
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("\n生成的JSON预览 (前3条记录):")
        preview_count = min(3, len(data))
        for i, (chain_id, chain) in enumerate(list(data.items())[:preview_count]):
            print(f"\n{chain_id}:")
            for key, value in chain.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main()