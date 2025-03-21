# coke_loader.py
"""
COKE认知链数据加载器
用于加载和处理认知链数据，支持从CSV和JSON导入
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Union
from pathlib import Path
import os

# 导入认知链模型
from cognitive_model import CognitiveChain

def load_cognitive_chains_from_json(file_path: str) -> Dict[str, CognitiveChain]:
    """从JSON文件加载认知链数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        认知链字典，键为认知链ID，值为CognitiveChain实例
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        chain_data = json.load(f)
    
    cognitive_chains = {}
    
    # 处理不同格式的JSON数据
    if isinstance(chain_data, dict):
        # 格式: {"chain_id": {...属性...}, ...}
        for chain_id, data in chain_data.items():
            cognitive_chains[chain_id] = CognitiveChain(**data)
    elif isinstance(chain_data, list):
        # 格式: [{...属性...}, {...属性...}, ...]
        for i, data in enumerate(chain_data):
            cognitive_chains[f"chain_{i}"] = CognitiveChain(**data)
    
    return cognitive_chains

def load_cognitive_chains_from_csv(
    directory_path: str,
    situation_file: Optional[str] = None,
    clue_file: Optional[str] = None,
    thought_file: Optional[str] = None,
    emotion_file: Optional[str] = None,
    action_file: Optional[str] = None,
    polarity_mapping: Dict[str, float] = None
) -> Dict[str, CognitiveChain]:
    """从CSV文件加载认知链数据
    
    Args:
        directory_path: CSV文件所在目录
        situation_file: 情境CSV文件名
        clue_file: 线索CSV文件名
        thought_file: 思维CSV文件名
        emotion_file: 情绪CSV文件名
        action_file: 行为CSV文件名
        polarity_mapping: 情绪类型到极性值的映射
        
    Returns:
        认知链字典，键为认知链ID，值为CognitiveChain实例
    """
    # 默认的情绪极性映射（可根据实际需要调整）
    default_polarity = {
        "高兴": 0.8, "开心": 0.9, "满足": 0.7, "兴奋": 0.9, 
        "自豪": 0.6, "感激": 0.7, "安心": 0.5, "平静": 0.3,
        "悲伤": -0.7, "难过": -0.8, "失望": -0.6, "沮丧": -0.7, 
        "生气": -0.8, "愤怒": -0.9, "恐惧": -0.8, "焦虑": -0.7,
        "羞愧": -0.6, "内疚": -0.6, "尴尬": -0.5, "无奈": -0.4
    }
    
    if polarity_mapping:
        default_polarity.update(polarity_mapping)
    
    # 确保文件路径正确
    base_path = Path(directory_path)
    
    # 加载各CSV文件
    dfs = {}
    
    if clue_file and os.path.exists(base_path / clue_file):
        dfs['clue'] = pd.read_csv(base_path / clue_file)
    
    if thought_file and os.path.exists(base_path / thought_file):
        dfs['thought'] = pd.read_csv(base_path / thought_file)
    
    if emotion_file and os.path.exists(base_path / emotion_file):
        dfs['emotion'] = pd.read_csv(base_path / emotion_file)
    
    if action_file and os.path.exists(base_path / action_file):
        dfs['action'] = pd.read_csv(base_path / action_file)
    
    if situation_file and os.path.exists(base_path / situation_file):
        dfs['situation'] = pd.read_csv(base_path / situation_file)
    elif 'clue' in dfs:
        # 如果没有专门的情境文件，使用线索作为情境
        dfs['situation'] = dfs['clue'].copy()
    
    # 确保所有数据帧的大小一致
    min_size = min(len(df) for df in dfs.values()) if dfs else 0
    
    # 构建认知链
    cognitive_chains = {}
    for i in range(min_size):
        # 从各数据帧提取对应行的数据
        chain_data = {}
        
        # 提取情境
        if 'situation' in dfs:
            chain_data['situation'] = str(dfs['situation'].iloc[i].iloc[0])
        
        # 提取线索
        if 'clue' in dfs:
            chain_data['clue'] = str(dfs['clue'].iloc[i].iloc[0])
        
        # 提取思维
        if 'thought' in dfs:
            chain_data['thought'] = str(dfs['thought'].iloc[i].iloc[0])
        
        # 提取情绪
        if 'emotion' in dfs:
            emotion = str(dfs['emotion'].iloc[i].iloc[0])
            chain_data['emotion'] = emotion
            # 根据情绪类型设置极性
            for emotion_type, polarity in default_polarity.items():
                if emotion_type in emotion:
                    chain_data['polarity'] = polarity
                    break
            else:
                chain_data['polarity'] = 0.0  # 默认极性为中性
        
        # 提取行为
        if 'action' in dfs:
            chain_data['action'] = str(dfs['action'].iloc[i].iloc[0])
        
        # 设置默认激活程度
        chain_data['activation_level'] = 0.5
        
        # 创建认知链实例
        cognitive_chains[f"chain_{i}"] = CognitiveChain(**chain_data)
    
    return cognitive_chains

def generate_sample_cognitive_chains() -> Dict[str, CognitiveChain]:
    """生成示例认知链数据，用于测试
    
    Returns:
        示例认知链字典
    """
    return {
        "chain_001": CognitiveChain(
            situation="被批评",
            clue="严厉语气",
            thought="我总是做不好任何事情",
            emotion="难过",
            action="沉默不语",
            polarity=-0.8,
            activation_level=0.5
        ),
        "chain_002": CognitiveChain(
            situation="得到表扬",
            clue="表扬工作成果",
            thought="我的付出得到了认可",
            emotion="高兴",
            action="微笑并感谢",
            polarity=0.9,
            activation_level=0.5
        ),
        "chain_003": CognitiveChain(
            situation="遇到困难",
            clue="任务复杂",
            thought="我无法应对这么困难的任务",
            emotion="焦虑",
            action="回避任务",
            polarity=-0.7,
            activation_level=0.5
        ),
        "chain_004": CognitiveChain(
            situation="完成目标",
            clue="达成目标",
            thought="我有能力实现我的目标",
            emotion="自豪",
            action="分享经验",
            polarity=0.8,
            activation_level=0.5
        )
    }

def save_cognitive_chains_to_json(cognitive_chains: Dict[str, CognitiveChain], file_path: str) -> None:
    """将认知链数据保存为JSON文件
    
    Args:
        cognitive_chains: 认知链字典
        file_path: 保存的文件路径
    """
    # 将认知链对象转换为字典
    chain_dict = {chain_id: chain.dict() for chain_id, chain in cognitive_chains.items()}
    
    # 保存为JSON文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(chain_dict, f, ensure_ascii=False, indent=4)

def create_cognitive_chains_from_csv_files(csv_dir='../data', output_json='cognitive_chains.json'):
    """从CSV文件整合创建认知链数据
    
    Args:
        csv_dir: CSV文件目录
        output_json: 输出的JSON文件路径
    """
    try:
        # 检查目录是否存在
        if not os.path.exists(csv_dir):
            print(f"目录{csv_dir}不存在!")
            return
            
        # 读取CSV文件
        clue_file = os.path.join(csv_dir, 'train_clue.csv')
        thought_file = os.path.join(csv_dir, 'train_thought.csv')
        emotion_file = os.path.join(csv_dir, 'train_emotion.csv')
        action_file = os.path.join(csv_dir, 'train_action.csv')
        
        # 检查文件是否存在
        files_exist = all(os.path.exists(f) for f in [clue_file, thought_file, emotion_file, action_file])
        if not files_exist:
            print("一个或多个CSV文件不存在!")
            return
            
        # 读取数据
        df_clue = pd.read_csv(clue_file)
        df_thought = pd.read_csv(thought_file)
        df_emotion = pd.read_csv(emotion_file)
        df_action = pd.read_csv(action_file)
        
        # 检查数据行数
        min_rows = min(len(df_clue), len(df_thought), len(df_emotion), len(df_action))
        if min_rows == 0:
            print("某个CSV文件不包含数据!")
            return
            
        # 创建合并数据帧
        df_combined = pd.DataFrame({
            'situation': df_clue.iloc[:min_rows, 0],  # 线索作为情境
            'clue': df_clue.iloc[:min_rows, 0],
            'thought': df_thought.iloc[:min_rows, 0],
            'emotion': df_emotion.iloc[:min_rows, 0],
            'action': df_action.iloc[:min_rows, 0],
        })
        
        # 添加极性和激活程度
        df_combined['polarity'] = 0.0
        df_combined['activation_level'] = 0.5
        
        # 设置极性值
        default_polarity = {
            "高兴": 0.8, "开心": 0.9, "满足": 0.7, "兴奋": 0.9, 
            "自豪": 0.6, "感激": 0.7, "安心": 0.5, "平静": 0.3,
            "悲伤": -0.7, "难过": -0.8, "失望": -0.6, "沮丧": -0.7, 
            "生气": -0.8, "愤怒": -0.9, "恐惧": -0.8, "焦虑": -0.7,
            "羞愧": -0.6, "内疚": -0.6, "尴尬": -0.5, "无奈": -0.4
        }
        
        # 应用情绪极性
        for i, emotion in enumerate(df_combined['emotion']):
            for emotion_type, polarity in default_polarity.items():
                if isinstance(emotion, str) and emotion_type in emotion:
                    df_combined.at[i, 'polarity'] = polarity
                    break
        
        # 将数据帧转换为JSON并保存
        records = df_combined.to_dict('records')
        cognitive_chains = {f"chain_{i}": record for i, record in enumerate(records)}
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(cognitive_chains, f, ensure_ascii=False, indent=4)
        
        print(f"成功创建认知链JSON数据，共{len(cognitive_chains)}条记录，保存到{output_json}")
        return True
    except Exception as e:
        print(f"创建认知链数据时发生错误：{e}")
        return False

def create_cognitive_chains_from_multiple_csv_files(
    thought_csv_path,
    emotion_csv_path,
    action_csv_path,
    output_json_path='integrated_cognitive_chains.json'
):
    """从多个CSV文件创建完整的认知链数据集
    
    Args:
        thought_csv_path: 思维CSV文件路径 (包含situation, clue, thought)
        emotion_csv_path: 情绪CSV文件路径 (包含situation, thought, emotion)
        action_csv_path: 行为CSV文件路径 (包含situation, thought, action)
        output_json_path: 输出JSON文件路径
        
    Returns:
        生成的认知链字典
    """
    # 加载基础数据 (Thought)
    df_thought = pd.read_csv(thought_csv_path)

    # 加载情绪数据 (Emotion)
    df_emotion = pd.read_csv(emotion_csv_path)

    # 加载行为数据 (Action)
    df_action = pd.read_csv(action_csv_path)

    # 整合数据（以情境与思维为主键）
    df_combined = df_thought.merge(df_emotion[['situation', 'thought', 'emotion', 'polarity']],
                               on=['situation', 'thought'], 
                               how='inner', 
                               suffixes=('', '_emotion'))

    df_combined = df_combined.merge(df_action[['situation', 'thought', 'action', 'polarity']],
                                on=['situation', 'thought'], 
                                how='inner', 
                                suffixes=('', '_action'))

    # 统计数据情况
    initial_records = len(df_thought)
    matched_records = len(df_combined)
    match_rate = (matched_records / initial_records) * 100 if initial_records > 0 else 0
    print(f"\n原始思维数据条数: {initial_records}")
    print(f"成功匹配的数据条数: {matched_records}")
    print(f"匹配率: {match_rate:.2f}%")
    
    # 选择最终字段
    df_final = df_combined[['situation', 'clue', 'thought', 'emotion', 'action', 'final_polarity']]
    df_final = df_final.rename(columns={'final_polarity': 'polarity'})
    df_final['activation_level'] = 0.5  # 默认激活程度

    # 处理缺失值
    df_final.fillna({'emotion': 'Unknown', 'action': 'Unknown', 'final_polarity': 0.0}, inplace=True)

    # 计算最终极性（简单策略：情绪和行为极性的均值）
    df_final['final_polarity'] = df_final[['polarity', 'polarity_action']].mean(axis=1)
    
    # 选择最终字段
    df_final = df_final[['situation', 'clue', 'thought', 'emotion', 'action', 'final_polarity']]
    df_final = df_final.rename(columns={'final_polarity': 'polarity'})
    df_final['activation_level'] = 0.5  # 默认激活程度

    # 创建认知链字典
    cognitive_chains = {}
    for i, row in df_final.iterrows():
        chain_id = f"chain_{i:03d}"
        cognitive_chains[chain_id] = CognitiveChain(
            situation=row['situation'],
            clue=row['clue'],
            thought=row['thought'],
            emotion=row['emotion'],
            action=row['action'],
            polarity=float(row['polarity']),
            activation_level=float(row['activation_level'])
        )
        
        # 每100个链条打印一次进度
        if i % 100 == 0:
            print(f"已处理 {i} 条认知链")
    
    # 保存为JSON文件
    if output_json_path:
        save_cognitive_chains_to_json(cognitive_chains, output_json_path)

    print(f"成功整合认知链数据，总计记录数: {len(cognitive_chains)}，已保存到 {output_json_path}")
    return cognitive_chains