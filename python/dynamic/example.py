# 文件路径: c:\Users\caogu\Downloads\patient-psi-main\python\dynamic\example.py

import os
import json
from datetime import datetime
import sys

# 添加python-dotenv用于加载环境变量
from dotenv import load_dotenv
import pathlib

# 获取项目根目录并加载.env.local文件
project_root = pathlib.Path(__file__).parent.parent.parent
env_path = project_root / '.env.local'
load_dotenv(dotenv_path=env_path)

# 添加父目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dynamic.cognitive_model import DynamicCognitiveModel, StateSnapshot, CoreBelief, Emotion
from dynamic.session_manager import TherapySessionManager, SessionConfig

def create_example_model():
    """创建示例患者模型"""
    # 创建初始状态快照
    initial_state = StateSnapshot(
        timestamp=datetime.now().isoformat(),
        trigger="initialization",
        core_beliefs={
            "helpless_1": CoreBelief(
                content="我无法控制自己的生活",
                type="helpless",
                strength=0.9,
                activation_level=0.8
            ),
            "unlovable_1": CoreBelief(
                content="我不值得被爱",
                type="unlovable",
                strength=0.8,
                activation_level=0.7
            )
        },
        intermediate_beliefs={},
        automatic_thoughts={},
        emotions={
            "emotion_1": Emotion(
                type="sadness",
                intensity=0.8
            ),
            "emotion_2": Emotion(
                type="anxiety",
                intensity=0.6
            )
        },
        behaviors={},
        coping_strategies={},
        depression_level=0.7
    )
    
    # 创建动态认知模型
    model = DynamicCognitiveModel(
        patient_id="example_patient_001",
        name="张明",
        history="32岁，软件工程师。近期因为工作压力大导致抑郁症状。童年经历过父母离异，与父亲关系紧张。",
        current_state=initial_state,
        state_history=[],
        conversation_history=[]
    )
    
    return model

def run_example_conversation():
    """运行示例对话"""
    # 创建示例模型
    model = create_example_model()
    
    # 配置会话
    config = SessionConfig(
        use_llm=True,  # 如果没有API密钥，设置为False
        track_progress=True,
        save_history=True,
        output_dir="./example_output"
    )
    
    # 创建会话管理器
    session = TherapySessionManager(model, config)
    
    # 示例对话
    conversation = [
        "今天你感觉怎么样？",
        "你能告诉我更多关于你工作中的压力吗？",
        "听起来你似乎对自己要求很高，是这样吗？",
        "你觉得这种想法对你有什么影响？",
        "我们可以一起探索一些应对压力的方法。你过去是如何处理压力的？"
    ]
    
    print("=== 开始示例对话 ===\n")
    
    for i, therapist_message in enumerate(conversation):
        print(f"治疗师 ({i+1}/{len(conversation)}): {therapist_message}")
        
        # 处理消息
        result = session.process_therapist_message(therapist_message)
        
        # 显示患者回复
        print(f"患者: {result['patient_response']}")
        
        # 显示认知更新摘要
        if result['cognitive_updates']:
            print("\n认知更新:")
            print(json.dumps(result['cognitive_updates'], ensure_ascii=False, indent=2))
        
        # 显示进度分析
        if result['progress_analysis'] and result['progress_analysis'].get('suggestions'):
            print("\n治疗建议:")
            for category, suggestions in result['progress_analysis']['suggestions'].items():
                for suggestion in suggestions:
                    print(f"- {suggestion}")
        
        print("\n" + "-"*50 + "\n")
    
    print("=== 对话结束 ===")
    print(f"模型状态已保存到: {config.output_dir}")

if __name__ == "__main__":
    print("确保你已设置OPENAI_API_KEY环境变量，或将SessionConfig中的use_llm设置为False。")
    run_example_conversation()