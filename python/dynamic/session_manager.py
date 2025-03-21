# 文件路径: c:\Users\caogu\Downloads\patient-psi-main\python\dynamic\session_manager.py

import os
from typing import Dict, Any, List, Optional, Tuple
import json
from datetime import datetime
from pydantic import BaseModel

# 导入其他组件
from .cognitive_model import DynamicCognitiveModel
from .updater import CognitiveUpdateEngine, SimpleUpdateEngine
from .response_generator import PatientResponseGenerator, SimpleResponseGenerator
from .tracker import TherapyProgressTracker

class SessionConfig(BaseModel):
    """治疗会话配置"""
    use_llm: bool = True  # 是否使用LLM进行更新和响应生成
    track_progress: bool = True  # 是否跟踪治疗进展
    save_history: bool = True  # 是否保存会话历史
    output_dir: str = "./output"  # 输出目录
    
    @classmethod
    def default(cls) -> "SessionConfig":
        """默认会话配置"""
        return cls()

class TherapySessionManager:
    """治疗会话管理器，协调认知模型、更新和响应生成"""
    
    def __init__(
        self, 
        model: DynamicCognitiveModel,
        config: Optional[SessionConfig] = None,
        api_key: Optional[str] = None
    ):
        """初始化会话管理器
        
        Args:
            model: 动态认知模型
            config: 会话配置
            api_key: OpenAI API密钥
        """
        self.model = model
        self.config = config or SessionConfig.default()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # 设置输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 初始化组件
        if self.config.use_llm:
            self.update_engine = CognitiveUpdateEngine(api_key=self.api_key)
            self.response_generator = PatientResponseGenerator(api_key=self.api_key)
        else:
            self.update_engine = SimpleUpdateEngine()
            self.response_generator = SimpleResponseGenerator()
        
        if self.config.track_progress:
            self.progress_tracker = TherapyProgressTracker(
                output_dir=os.path.join(self.config.output_dir, "reports")
            )
        else:
            self.progress_tracker = None
    
    def process_therapist_message(self, therapist_message: str) -> Dict[str, Any]:
        """处理治疗师消息，更新模型并生成回复
        
        Args:
            therapist_message: 治疗师消息
            
        Returns:
            包含患者回复和分析结果的字典
        """
        # 更新认知模型
        if isinstance(self.update_engine, CognitiveUpdateEngine):
            updated_model, updates = self.update_engine.update_model(
                self.model, therapist_message
            )
        else:
            updates = self.update_engine.compute_updates(self.model, therapist_message)
            self.model.update_state(updates, f"therapist_message:{datetime.now().isoformat()}")
            self.model.add_message_to_history("therapist", therapist_message)
            updated_model = self.model
        
        # 生成患者回复
        patient_response = self.response_generator.generate_response(
            updated_model, therapist_message
        )
        
        # 如果启用了进度跟踪，更新进度
        progress_analysis = None
        if self.progress_tracker:
            progress_analysis = self.progress_tracker.analyze_progress(updated_model)
        
        # 如果启用了历史保存，保存当前状态
        if self.config.save_history:
            self._save_session_state()
        
        return {
            "patient_response": patient_response,
            "cognitive_updates": updates,
            "progress_analysis": progress_analysis,
            "model_state": {
                "depression_level": updated_model.current_state.depression_level,
                "session_count": len(updated_model.state_history) + 1
            }
        }
    
    def _save_session_state(self):
        """保存当前会话状态"""
        # 创建会话保存目录
        session_dir = os.path.join(
            self.config.output_dir, 
            "sessions", 
            self.model.patient_id
        )
        os.makedirs(session_dir, exist_ok=True)
        
        # 保存时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型状态
        model_file = os.path.join(session_dir, f"model_state_{timestamp}.json")
        self.model.save_to_file(model_file)
        
        return model_file
    
    @classmethod
    def load_session(
        cls, 
        model_file: str, 
        config: Optional[SessionConfig] = None,
        api_key: Optional[str] = None
    ) -> "TherapySessionManager":
        """从文件加载会话
        
        Args:
            model_file: 模型状态文件路径
            config: 会话配置
            api_key: OpenAI API密钥
            
        Returns:
            会话管理器实例
        """
        # 加载模型
        model = DynamicCognitiveModel.load_from_file(model_file)
        
        # 创建会话管理器
        return cls(model, config, api_key)