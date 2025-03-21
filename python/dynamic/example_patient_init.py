# example_patient_init.py
"""
示例患者初始化脚本
演示如何加载认知链数据并初始化动态认知模型
"""

import os
import json
import pandas as pd
from cognitive_model import DynamicCognitiveModel, CognitiveChain
from coke_loader import (
    load_cognitive_chains_from_json, 
    load_cognitive_chains_from_csv,
    generate_sample_cognitive_chains,
    save_cognitive_chains_to_json,
    create_cognitive_chains_from_csv_files,
    create_cognitive_chains_from_multiple_csv_files
)

def init_from_static_json():
    """从静态JSON文件初始化动态模型"""
    # 加载预定义的静态模型
    try:
        with open('static_patient_model.json', 'r', encoding='utf-8') as f:
            static_model = json.load(f)
    except FileNotFoundError:
        print("静态模型文件未找到，将使用默认值")
        static_model = {
            "name": "示例患者",
            "age": 25,
            "gender": "female",
            "background": "大学生，有轻度抑郁症状",
            "diagnosis": "轻度抑郁",
            "helpless_belief": [
                "我无法控制我的情绪",
                "我的问题无法得到解决"
            ],
            "unlovable_belief": [
                "我不值得被爱",
                "我不被他人接受"
            ],
            "emotion": [
                "抑郁",
                "焦虑",
                "内疚"
            ],
            "behavior": "逃避行为，社交退缩，消极应对",
            "triggers": [
                "批评", "比较", "失败", "拒绝"
            ]
        }
    
    # 生成或加载认知链数据
    if os.path.exists('cognitive_chains.json'):
        print("从现有JSON文件加载认知链数据")
        cognitive_chains = load_cognitive_chains_from_json('cognitive_chains.json')
    else:
        print("未找到认知链JSON文件，尝试从CSV文件创建")
        
        if os.path.exists('../data'):
            # 从CSV文件创建认知链数据
            success = create_cognitive_chains_from_csv_files('../data', 'cognitive_chains.json')
            if success:
                cognitive_chains = load_cognitive_chains_from_json('cognitive_chains.json')
            else:
                print("无法从CSV文件创建认知链，将使用示例数据")
                cognitive_chains = generate_sample_cognitive_chains()
                save_cognitive_chains_to_json(cognitive_chains, 'cognitive_chains.json')
        else:
            print("未找到数据目录，使用示例认知链数据")
            cognitive_chains = generate_sample_cognitive_chains()
            save_cognitive_chains_to_json(cognitive_chains, 'cognitive_chains.json')
    
    # 初始化动态认知模型
    dynamic_model = DynamicCognitiveModel.from_static_model(static_model, cognitive_chains)
    
    print(f"动态认知模型初始化完成")
    print(f"模型包含以下信息:")
    print(f"- 患者名称: {dynamic_model.name}")
    print(f"- 诊断: {static_model.get('diagnosis', '未指定')}")
    print(f"- 核心信念数量: {len(dynamic_model.current_state.core_beliefs)}")
    print(f"- 行为数量: {len(dynamic_model.current_state.behaviors)}")
    print(f"- 情绪数量: {len(dynamic_model.current_state.emotions)}")
    print(f"- 认知链数量: {len(dynamic_model.current_state.cognitive_chains)}")
    
    # 打印几个示例认知链
    print("\n认知链示例:")
    for i, (chain_id, chain) in enumerate(dynamic_model.current_state.cognitive_chains.items()):
        if i >= 3:  # 只打印前3个
            break
        print(f"\n链ID: {chain_id}")
        print(f"情境: {chain.situation}")
        print(f"线索: {chain.clue}")
        print(f"思维: {chain.thought}")
        print(f"情绪: {chain.emotion}")
        print(f"行为: {chain.action}")
        print(f"极性: {chain.polarity}")
    
    return dynamic_model

def init_from_english_data():
    """从英文CSV数据集整合并初始化模型"""
    print("\n正在从CSV文件整合英文数据...")
    
    # 设置数据文件路径
    thought_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'train_thought.csv')
    emotion_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'train_emotion.csv')
    action_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'train_action.csv')
    integrated_json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'integrated_cognitive_chains.json')
    
    # 检查文件是否存在
    all_files_exist = True
    for path in [thought_csv_path, emotion_csv_path, action_csv_path]:
        if not os.path.exists(path):
            print(f"\n错误: 无法找到数据文件 {path}")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n无法整合数据: 文件不存在")
        return None
        
    # 设置样本大小
    sample_size = 1000  # 可以设置为更大的值或None以使用全部数据
    
    try:
        # 加载数据
        df_thought = pd.read_csv(thought_csv_path)
        if sample_size:
            df_thought = df_thought.sample(sample_size, random_state=42)
            # 保存样本数据 - 使用绝对路径
            sample_thought_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'sample_thought.csv')
            # 确保目录存在
            os.makedirs(os.path.dirname(sample_thought_path), exist_ok=True)
            df_thought.to_csv(sample_thought_path, index=False)
            thought_csv_path = sample_thought_path
            print(f"\n使用 {sample_size} 条样本数据进行处理")
        
        # 整合数据
        print("\n开始整合数据...")
        # 创建保存示例数据的目录
        os.makedirs(os.path.dirname(integrated_json_path), exist_ok=True)
        cognitive_chains = create_cognitive_chains_from_multiple_csv_files(
            thought_csv_path,
            emotion_csv_path,
            action_csv_path,
            integrated_json_path
        )
        
        # 初始化静态模型
        static_model = {
            "name": "English Patient",
            "age": 30,
            "gender": "female",
            "background": "College student with mild depression",
            "diagnosis": "Mild Depression",
            "helpless_belief": [
                "I cannot control my emotions",
                "My problems cannot be solved"
            ],
            "unlovable_belief": [
                "I am not worthy of being loved",
                "I am not accepted by others"
            ],
            "emotion": [
                "Depressed",
                "Anxious",
                "Guilty"
            ],
            "behavior": "Avoidance behavior, social withdrawal, negative coping",
            "triggers": [
                "Criticism", "Comparison", "Failure", "Rejection"
            ]
        }
        
        # 初始化动态模型
        print("\n使用整合数据初始化动态模型...")
        dynamic_model = DynamicCognitiveModel.from_static_model(static_model, cognitive_chains)
        
        # 打印模型信息
        print("\n模型初始化完成")
        print(f"模型包含以下信息:")
        print(f"- 患者名称: {dynamic_model.name}")
        print(f"- 诊断: {static_model.get('diagnosis', '未指定')}")
        print(f"- 核心信念数量: {len(dynamic_model.current_state.core_beliefs)}")
        print(f"- 行为数量: {len(dynamic_model.current_state.behaviors)}")
        print(f"- 情绪数量: {len(dynamic_model.current_state.emotions)}")
        print(f"- 认知链数量: {len(dynamic_model.current_state.cognitive_chains)}")
        
        # 打印几个示例认知链
        print("\n认知链示例:")
        sample_chains = list(dynamic_model.current_state.cognitive_chains.items())[:3]
        for chain_id, chain in sample_chains:
            print(f"\n链ID: {chain_id}")
            print(f"情境: {chain.situation}")
            print(f"线索: {chain.clue}")
            print(f"思维: {chain.thought}")
            print(f"情绪: {chain.emotion}")
            print(f"行为: {chain.action}")
            print(f"极性: {chain.polarity}")
        
        return dynamic_model
    except Exception as e:
        print(f"\n处理英文数据时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def init_from_csv_files():
    """从CSV文件加载认知链并初始化"""
    print("\n从CSV文件加载认知链...")
    
    # 设置数据文件路径
    csv_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'cognitive_chains.json')
    
    try:
        # 检查文件是否存在
        if os.path.exists(output_path):
            print(f"\n发现已存在的JSON文件: {output_path}")
            cognitive_chains = load_cognitive_chains_from_json(output_path)
            print(f"\n加载已存在的认知链数据: {len(cognitive_chains)} 条")
        else:
            # 如果没有JSON文件，尝试从CSV文件创建
            print("\n尝试从CSV文件创建认知链...")
            csv_created = create_cognitive_chains_from_csv_files(csv_dir, output_path)
            
            if csv_created:
                cognitive_chains = load_cognitive_chains_from_json(output_path)
                print(f"\n成功从CSV文件创建认知链数据: {len(cognitive_chains)} 条")
            else:
                print("\n无法从CSV文件创建认知链，将使用示例数据")
                cognitive_chains = generate_sample_cognitive_chains()
                save_cognitive_chains_to_json(cognitive_chains, output_path)
                print(f"\n使用示例数据创建认知链: {len(cognitive_chains)} 条")
        
        # 初始化静态模型
        static_model = {
            "name": "示例患者",
            "helpless_belief": ["我无法控制我的情绪", "我的问题无法得到解决"],
            "unlovable_belief": ["我不值得被爱", "我不被他人接受"],
            "emotion": ["抑郁", "焦虑", "内疚"],
            "behavior": "逃避行为，社交退缩，消极应对",
            "diagnosis": "轻度抑郁"
        }
        
        # 初始化动态模型
        dynamic_model = DynamicCognitiveModel.from_static_model(static_model, cognitive_chains)
        
        # 打印模型信息
        print("\n模型初始化完成")
        print(f"模型包含以下信息:")
        print(f"- 患者名称: {dynamic_model.name}")
        print(f"- 诊断: {static_model.get('diagnosis', '未指定')}")
        print(f"- 核心信念数量: {len(dynamic_model.current_state.core_beliefs)}")
        print(f"- 行为数量: {len(dynamic_model.current_state.behaviors)}")
        print(f"- 情绪数量: {len(dynamic_model.current_state.emotions)}")
        print(f"- 认知链数量: {len(dynamic_model.current_state.cognitive_chains)}")
        
        # 打印几个示例认知链
        print("\n认知链示例:")
        sample_chains = list(dynamic_model.current_state.cognitive_chains.items())[:3]
        for chain_id, chain in sample_chains:
            print(f"\n链ID: {chain_id}")
            print(f"情境: {chain.situation}")
            print(f"线索: {chain.clue}")
            print(f"思维: {chain.thought}")
            print(f"情绪: {chain.emotion}")
            print(f"行为: {chain.action}")
            print(f"极性: {chain.polarity}")
        
        return dynamic_model
    except Exception as e:
        print(f"\n处理CSV文件时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_and_save_sample_chains():
    """生成并保存示例认知链数据"""
    print("\n生成示例认知链数据...")
    cognitive_chains = generate_sample_cognitive_chains()
    
    # 设置输出文件路径
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'cognitive_chains.json')
    
    # 保存数据
    save_cognitive_chains_to_json(cognitive_chains, output_path)
    print(f"\n成功生成并保存示例认知链数据: {len(cognitive_chains)} 条")
    
    return cognitive_chains

def main():
    """主函数"""
    # 选择初始化方法
    print("请选择初始化方法:")
    print("1. 从静态JSON文件初始化")
    print("2. 从CSV文件加载认知链并初始化")
    print("3. 创建并保存示例认知链数据")
    print("4. 从英文CSV数据集整合并初始化模型")
    
    choice = input("请输入选择(1-4): ")
    
    if choice == "1":
        dynamic_model = init_from_static_json()
    elif choice == "2": 
        dynamic_model = init_from_csv_files()
    elif choice == "3":
        generate_and_save_sample_chains()
        return
    elif choice == "4":
        dynamic_model = init_from_english_data()
    else:
        print("无效选择!")
        return

if __name__ == "__main__":
    main()