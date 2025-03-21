from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import json
import copy
import os

# 基础模型组件
class CoreBelief(BaseModel):
    """核心信念模型"""
    content: str
    strength: float = Field(default=1.0, ge=0.0, le=1.0)  # 信念强度 (0-1)
    type: str = Field(default="helpless")  # helpless, unlovable, worthless
    activation_level: float = Field(default=0.5, ge=0.0, le=1.0)  # 当前激活程度
    
class IntermediateBelief(BaseModel):
    """中间信念模型"""
    content: str
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    type: str = Field(default="attitude")  # attitude, rule, assumption
    activation_level: float = Field(default=0.5, ge=0.0, le=1.0)
    
class AutomaticThought(BaseModel):
    """自动思维模型"""
    content: str
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    situation: Optional[str] = None  # 触发情境
    
class Emotion(BaseModel):
    """情绪模型"""
    type: str  # 情绪类型
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)  # 情绪强度
    
class Behavior(BaseModel):
    """行为模型"""
    description: str
    frequency: float = Field(default=0.5, ge=0.0, le=1.0)  # 行为频率
    
class CopingStrategy(BaseModel):
    """应对策略模型"""
    description: str
    effectiveness: float = Field(default=0.5, ge=0.0, le=1.0)  # 有效性
    usage_frequency: float = Field(default=0.5, ge=0.0, le=1.0)  # 使用频率

# 认知链结构（基于COKE框架）
class CognitiveChain(BaseModel):
    """认知链模型，基于COKE框架，表示情境-思维-情绪-行为链条"""
    situation: str = Field(description="触发情境")  # 情境
    clue: Optional[str] = Field(default=None, description="情境中的关键线索")  # 情境关键线索
    thought: str = Field(description="产生的思维内容")  # 产生的具体思维
    emotion: str = Field(description="关联的情绪反应")  # 相应情绪
    action: str = Field(description="后续采取的行为")  # 后续行为
    polarity: float = Field(default=0.0, ge=-1.0, le=1.0, description="情绪极性，-1.0为极度消极，+1.0为极度积极")  # 情绪极性
    activation_level: float = Field(default=0.5, ge=0.0, le=1.0, description="当前激活程度")  # 激活程度

# 认知状态快照
class StateSnapshot(BaseModel):
    """认知状态快照"""
    timestamp: str
    trigger: str  # 触发状态变化的事件
    core_beliefs: Dict[str, CoreBelief]
    intermediate_beliefs: Dict[str, IntermediateBelief]
    automatic_thoughts: Dict[str, AutomaticThought]
    emotions: Dict[str, Emotion]
    behaviors: Dict[str, Behavior]
    coping_strategies: Dict[str, CopingStrategy]
    depression_level: float = Field(default=0.5, ge=0.0, le=1.0)
    # 新增：认知链存储
    cognitive_chains: Dict[str, CognitiveChain] = {}
    
# 动态认知模型
class DynamicCognitiveModel(BaseModel):
    """动态认知模型，包含状态历史与更新方法"""
    patient_id: str
    name: str
    history: str  # 患者生活历史背景
    current_state: StateSnapshot  # 当前状态
    state_history: List[StateSnapshot] = []  # 状态历史记录
    conversation_history: List[Dict[str, str]] = []  # 对话历史
    
    def update_state(self, updates: Dict[str, Any], trigger: str) -> StateSnapshot:
        """更新认知状态并记录历史"""
        # 保存当前状态到历史
        self.state_history.append(copy.deepcopy(self.current_state))
        
        # 创建新状态为当前状态的深拷贝
        new_state = copy.deepcopy(self.current_state)
        new_state.timestamp = datetime.now().isoformat()
        new_state.trigger = trigger
        
        # 应用更新
        self._apply_updates(new_state, updates)
        
        # 更新当前状态
        self.current_state = new_state
        return self.current_state
    
    def _apply_updates(self, state: StateSnapshot, updates: Dict[str, Any], update_factor: float = 0.3):
        """递增式应用更新到状态"""
        # 更新抑郁水平
        if 'depression_level' in updates:
            current = state.depression_level
            target = updates['depression_level']
            state.depression_level = current * (1-update_factor) + target * update_factor
        
        # 更新核心信念
        if 'core_beliefs' in updates:
            for belief_id, update in updates['core_beliefs'].items():
                if belief_id in state.core_beliefs:
                    # 更新现有信念
                    belief = state.core_beliefs[belief_id]
                    for key, value in update.items():
                        if key == 'activation_level' and hasattr(belief, key):
                            # 递增式更新数值属性
                            current = getattr(belief, key)
                            setattr(belief, key, current * (1-update_factor) + value * update_factor)
                        elif hasattr(belief, key):
                            # 直接更新其他属性
                            setattr(belief, key, value)
                else:
                    # 添加新信念
                    state.core_beliefs[belief_id] = CoreBelief(**update)
        
        # 类似地更新其他部分
        # 更新中间信念
        if 'intermediate_beliefs' in updates:
            self._update_component(state.intermediate_beliefs, updates['intermediate_beliefs'], 
                                IntermediateBelief, update_factor)
            
        # 更新自动思维
        if 'automatic_thoughts' in updates:
            self._update_component(state.automatic_thoughts, updates['automatic_thoughts'], 
                              AutomaticThought, update_factor)
            
        # 更新情绪
        if 'emotions' in updates:
            self._update_component(state.emotions, updates['emotions'], 
                          Emotion, update_factor)
            
        # 更新行为
        if 'behaviors' in updates:
            self._update_component(state.behaviors, updates['behaviors'], 
                           Behavior, update_factor)
            
        # 更新应对策略
        if 'coping_strategies' in updates:
            self._update_component(state.coping_strategies, updates['coping_strategies'], 
                                CopingStrategy, update_factor)
        
        # 更新认知链
        if 'cognitive_chains' in updates:
            for chain_id, update in updates['cognitive_chains'].items():
                if chain_id in state.cognitive_chains:
                    chain = state.cognitive_chains[chain_id]
                    for key, value in update.items():
                        if isinstance(value, (int, float)) and hasattr(chain, key):
                            current = getattr(chain, key)
                            setattr(chain, key, current * (1-update_factor) + value * update_factor)
                        elif hasattr(chain, key):
                            setattr(chain, key, value)
                else:
                    state.cognitive_chains[chain_id] = CognitiveChain(**update)
    
    def _update_component(self, component_dict, updates, component_class, update_factor):
        """通用组件更新方法"""
        for item_id, update in updates.items():
            if item_id in component_dict:
                # 更新现有项
                item = component_dict[item_id]
                for key, value in update.items():
                    if isinstance(value, (int, float)) and hasattr(item, key):
                        # 递增式更新数值
                        current = getattr(item, key)
                        setattr(item, key, current * (1-update_factor) + value * update_factor)
                    elif hasattr(item, key):
                        # 直接更新其他属性
                        setattr(item, key, value)
            else:
                # 添加新项
                component_dict[item_id] = component_class(**update)
    
    def add_message_to_history(self, role: str, content: str):
        """添加消息到会话历史"""
        self.conversation_history.append({
            "role": role, 
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_recent_conversation(self, n: int = 5) -> List[Dict[str, str]]:
        """获取最近n条会话记录"""
        return self.conversation_history[-n:] if len(self.conversation_history) >= n else self.conversation_history
    
    def save_to_file(self, filepath: str):
        """保存模型到文件"""
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, indent=2, ensure_ascii=False)
            
        return f"模型已保存到 {filepath}"
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "DynamicCognitiveModel":
        """从文件加载模型"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def from_static_model(cls, static_model: Dict[str, Any], cognitive_chains: Optional[Dict[str, CognitiveChain]] = None):
        """从静态认知模型转换为动态认知模型
        
        Args:
            static_model: 静态认知模型数据
            cognitive_chains: 可选的认知链字典，用于初始化模型的认知链
            
        Returns:
            动态认知模型实例
        """
        # 核心信念
        core_beliefs = {}
        if 'helpless_belief' in static_model:
            for i, belief in enumerate(static_model['helpless_belief']):
                core_beliefs[f"helpless_{i}"] = CoreBelief(
                    content=belief,
                    type="helpless",
                    strength=1.0,
                    activation_level=0.8
                )
                
        if 'unlovable_belief' in static_model:
            for i, belief in enumerate(static_model['unlovable_belief']):
                core_beliefs[f"unlovable_{i}"] = CoreBelief(
                    content=belief,
                    type="unlovable",
                    strength=1.0,
                    activation_level=0.8
                )
                
        if 'worthless_belief' in static_model:
            for i, belief in enumerate(static_model['worthless_belief']):
                core_beliefs[f"worthless_{i}"] = CoreBelief(
                    content=belief,
                    type="worthless",
                    strength=1.0,
                    activation_level=0.8
                )
        
        # 解析中间信念
        intermediate_beliefs = {}
        if 'intermediate_belief' in static_model:
            # 分割中间信念文本
            beliefs_text = static_model['intermediate_belief']
            beliefs_parts = beliefs_text.split('[during depression]')
            
            # 常规中间信念
            regular_beliefs = beliefs_parts[0].strip().split('\n')
            for i, belief in enumerate(regular_beliefs):
                if belief.strip():
                    intermediate_beliefs[f"regular_{i}"] = IntermediateBelief(
                        content=belief.strip(),
                        type="attitude",
                        strength=1.0,
                        activation_level=0.7
                    )
            
            # 抑郁期间的中间信念
            if len(beliefs_parts) > 1:
                depression_beliefs = beliefs_parts[1].strip().split('\n')
                for i, belief in enumerate(depression_beliefs):
                    if belief.strip():
                        intermediate_beliefs[f"depression_{i}"] = IntermediateBelief(
                            content=belief.strip(),
                            type="rule",
                            strength=1.0,
                            activation_level=0.9 # 抑郁期间更活跃
                        )
        
        # 自动思维
        automatic_thoughts = {}
        if 'auto_thought' in static_model:
            auto_thought = static_model['auto_thought']
            situation = static_model.get('situation', '')
            automatic_thoughts["main"] = AutomaticThought(
                content=auto_thought,
                situation=situation,
                strength=0.9
            )
        
        # 情绪
        emotions = {}
        if 'emotion' in static_model and isinstance(static_model['emotion'], list):
            for i, emotion_text in enumerate(static_model['emotion']):
                emotion_type = emotion_text.strip()
                emotions[f"emotion_{i}"] = Emotion(
                    type=emotion_type,
                    intensity=0.8
                )
        
        # 行为
        behaviors = {}
        if 'behavior' in static_model:
            behavior_text = static_model['behavior']
            behaviors["main"] = Behavior(
                description=behavior_text,
                frequency=0.8
            )
        
        # 应对策略
        coping_strategies = {}
        if 'coping_strategies' in static_model:
            strategies_text = static_model['coping_strategies']
            strategies = strategies_text.split('.')
            for i, strategy in enumerate(strategies):
                if strategy.strip():
                    coping_strategies[f"strategy_{i}"] = CopingStrategy(
                        description=strategy.strip(),
                        effectiveness=0.3,  # 假设较低有效性
                        usage_frequency=0.7  # 假设较高使用频率
                    )
        
        # 创建初始状态快照
        initial_state = StateSnapshot(
            timestamp=datetime.now().isoformat(),
            trigger="initialization",
            core_beliefs=core_beliefs,
            intermediate_beliefs=intermediate_beliefs,
            automatic_thoughts=automatic_thoughts,
            emotions=emotions,
            behaviors=behaviors,
            coping_strategies=coping_strategies,
            depression_level=0.7,  # 假设初始抑郁水平
            cognitive_chains=cognitive_chains or {}
        )
        
        # 创建动态认知模型
        return cls(
            patient_id=static_model.get('id', 'unknown'),
            name=static_model.get('name', 'Patient'),
            history=static_model.get('history', ''),
            current_state=initial_state,
            state_history=[],
            conversation_history=[]
        )