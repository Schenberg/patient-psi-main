import os
import json
from typing import Dict, List, Any, Optional, Tuple
import openai
from pydantic import BaseModel
from datetime import datetime
import random

# 导入动态认知模型
from .cognitive_model import DynamicCognitiveModel, StateSnapshot

class ResponsePrompt(BaseModel):
    """用于生成患者响应的提示模板"""
    system_prompt: str
    user_prompt_template: str
    
    @classmethod
    def default(cls) -> "ResponsePrompt":
        """创建默认的响应提示模板"""
        system_prompt = """
        你是一个模拟认知行为治疗中的患者。你需要根据提供的患者认知状态信息，生成自然、真实的对话响应。
        你的响应应该反映患者当前的认知模式、情绪状态和应对策略，特别是那些激活度较高的核心信念和情绪。
        请确保响应风格一致，并适当表现出患者的疑虑、希望或抗拒，具体取决于当前的认知状态。
        """
        
        user_prompt_template = """
        患者信息：
        - 姓名: {name}
        - 背景: {history}
        
        当前认知状态：
        1. 核心信念 (按激活度排序): 
        {core_beliefs}
        
        2. 中间信念 (按激活度排序): 
        {intermediate_beliefs}
        
        3. 自动思维: 
        {automatic_thoughts}
        
        4. 主要情绪 (按强度排序): 
        {emotions}
        
        5. 典型行为: 
        {behaviors}
        
        6. 常用应对策略: 
        {coping_strategies}
        
        7. 抑郁水平: {depression_level} (0-1，数值越高表示抑郁程度越深)
        
        最近的对话历史 (最新的在最后):
        {conversation_history}
        
        治疗师最新消息: 
        "{therapist_message}"
        
        基于以上信息，生成一个自然、真实的患者回复。回复应该反映患者的核心信念和当前情绪状态。
        只需提供患者的直接回复，不要加入任何额外说明、引号或角色标识。
        """
        
        return cls(system_prompt=system_prompt, user_prompt_template=user_prompt_template)


class PatientResponseGenerator:
    """患者回复生成器，基于当前认知状态生成自然的回复"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """初始化回复生成器
        
        Args:
            api_key: OpenAI API密钥，如果为None则尝试从环境变量获取
            model: 使用的OpenAI模型名称
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供，请通过参数传入或设置OPENAI_API_KEY环境变量")
        
        self.model = model
        self.response_prompt = ResponsePrompt.default()
        openai.api_key = self.api_key
    
    def generate_response(self, model: DynamicCognitiveModel, therapist_message: str) -> str:
        """基于认知模型状态生成患者回复
        
        Args:
            model: 动态认知模型
            therapist_message: 治疗师消息
            
        Returns:
            生成的患者回复
        """
        # 准备提示数据
        prompt_data = self._prepare_prompt_data(model, therapist_message)
        
        # 调用LLM生成回复
        response = self._generate_with_llm(prompt_data)
        
        # 将回复添加到对话历史
        if response:
            model.add_message_to_history("patient", response)
            
        return response
    
    def _prepare_prompt_data(self, model: DynamicCognitiveModel, therapist_message: str) -> Dict[str, Any]:
        """准备提示数据
        
        Args:
            model: 动态认知模型
            therapist_message: 治疗师消息
            
        Returns:
            用于填充提示模板的数据字典
        """
        # 获取当前状态
        state = model.current_state
        
        # 对核心信念按激活度排序
        sorted_core_beliefs = sorted(
            state.core_beliefs.items(), 
            key=lambda x: x[1].activation_level, 
            reverse=True
        )
        core_beliefs_text = "\n".join([
            f"- {belief.content} (类型: {belief.type}, 激活度: {belief.activation_level:.2f})" 
            for _, belief in sorted_core_beliefs
        ])
        
        # 对中间信念按激活度排序
        sorted_intermediate_beliefs = sorted(
            state.intermediate_beliefs.items(), 
            key=lambda x: x[1].activation_level, 
            reverse=True
        )
        intermediate_beliefs_text = "\n".join([
            f"- {belief.content} (类型: {belief.type}, 激活度: {belief.activation_level:.2f})" 
            for _, belief in sorted_intermediate_beliefs
        ])
        
        # 自动思维
        automatic_thoughts_text = "\n".join([
            f"- {thought.content} (强度: {thought.strength:.2f})" 
            for _, thought in state.automatic_thoughts.items()
        ])
        
        # 对情绪按强度排序
        sorted_emotions = sorted(
            state.emotions.items(), 
            key=lambda x: x[1].intensity, 
            reverse=True
        )
        emotions_text = "\n".join([
            f"- {emotion.type} (强度: {emotion.intensity:.2f})" 
            for _, emotion in sorted_emotions
        ])
        
        # 行为
        behaviors_text = "\n".join([
            f"- {behavior.description} (频率: {behavior.frequency:.2f})" 
            for _, behavior in state.behaviors.items()
        ])
        
        # 应对策略
        coping_strategies_text = "\n".join([
            f"- {strategy.description} (有效性: {strategy.effectiveness:.2f}, 使用频率: {strategy.usage_frequency:.2f})" 
            for _, strategy in state.coping_strategies.items()
        ])
        
        # 对话历史
        conversation_history = model.get_recent_conversation(5)
        conversation_history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in conversation_history
        ])
        
        # 返回填充数据
        return {
            "name": model.name,
            "history": model.history,
            "core_beliefs": core_beliefs_text,
            "intermediate_beliefs": intermediate_beliefs_text,
            "automatic_thoughts": automatic_thoughts_text,
            "emotions": emotions_text,
            "behaviors": behaviors_text,
            "coping_strategies": coping_strategies_text,
            "depression_level": state.depression_level,
            "conversation_history": conversation_history_text,
            "therapist_message": therapist_message
        }
    
    def _generate_with_llm(self, prompt_data: Dict[str, Any]) -> str:
        """使用LLM生成患者回复
        
        Args:
            prompt_data: 模板填充数据
            
        Returns:
            生成的患者回复
        """
        try:
            # 填充用户提示模板
            user_prompt = self.response_prompt.user_prompt_template.format(**prompt_data)
            
            # 准备消息
            messages = [
                {"role": "system", "content": self.response_prompt.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,  # 稍高的temperature增加回复的多样性
                max_tokens=500
            )
            
            # 提取结果
            result = response.choices[0].message.content.strip()
            return result
            
        except Exception as e:
            print(f"生成回复时发生错误: {str(e)}")
            # 返回备用回复
            return self._generate_fallback_response(prompt_data)
    
    def _generate_fallback_response(self, prompt_data: Dict[str, Any]) -> str:
        """当LLM调用失败时生成备用回复
        
        Args:
            prompt_data: 模板填充数据
            
        Returns:
            备用回复
        """
        # 提取当前情绪作为备用回复的基础
        emotions_text = prompt_data.get("emotions", "")
        depression_level = prompt_data.get("depression_level", 0.5)
        
        # 基础响应模板
        templates = [
            "我不确定我现在该说什么...",
            "这很难解释...",
            "我需要再想想这个问题。",
            "我不知道该怎么回答。",
            "可以给我点时间思考一下吗？"
        ]
        
        # 如果抑郁水平较高，添加更多消极回复
        if depression_level > 0.6:
            templates.extend([
                "我感觉今天没什么用处。",
                "我不确定这能有什么帮助。",
                "有时候我觉得这一切都毫无意义。",
                "我尝试了，但是我不确定有什么变化。"
            ])
        
        # 随机选择一个响应
        return random.choice(templates)


class SimpleResponseGenerator:
    """简单回复生成器，基于模板和规则生成患者回复"""
    
    def __init__(self):
        """初始化简单回复生成器"""
        # 定义不同情绪状态的回复模板
        self.response_templates = {
            "depressed": [
                "我感觉很低落... {belief}",
                "我不知道该怎么办了... {belief}",
                "最近感觉什么都做不好... {belief}",
                "我真的很累，什么都没动力去做... {belief}",
                "我感觉自己就是个负担... {belief}"
            ],
            "anxious": [
                "我一直很担心... {belief}",
                "我感觉紧张，心跳加速... {belief}",
                "我无法控制这些担忧的想法... {belief}",
                "我总是在想最坏的情况... {belief}",
                "我觉得很不安全... {belief}"
            ],
            "angry": [
                "我感到很沮丧... {belief}",
                "我对这种情况感到厌烦... {belief}",
                "我控制不了我的脾气... {belief}",
                "有时候我真的很生气... {belief}",
                "所有事情都让我恼火... {belief}"
            ],
            "neutral": [
                "我不确定该怎么感觉... {belief}",
                "我在努力理解这些... {belief}",
                "我想知道这是否真的有用... {belief}",
                "我在尝试改变我的想法... {belief}",
                "我不确定这是否是个问题... {belief}"
            ],
            "hopeful": [
                "我想我可能看到了一些改善... {belief}",
                "也许情况会变好... {belief}",
                "我在尝试从不同角度看问题... {belief}",
                "我开始认识到一些事情... {belief}",
                "有时候我能感受到希望... {belief}"
            ]
        }
        
        # 定义情绪关键词映射
        self.emotion_mapping = {
            "sad": "depressed",
            "depressed": "depressed",
            "down": "depressed",
            "miserable": "depressed",
            "hopeless": "depressed",
            "anxious": "anxious",
            "nervous": "anxious",
            "worried": "anxious",
            "fearful": "anxious",
            "panicked": "anxious",
            "angry": "angry",
            "frustrated": "angry",
            "irritated": "angry",
            "annoyed": "angry",
            "resentful": "angry",
            "content": "hopeful",
            "hopeful": "hopeful",
            "optimistic": "hopeful",
            "encouraged": "hopeful",
            "positive": "hopeful"
        }
    
    def generate_response(self, model: DynamicCognitiveModel, therapist_message: str) -> str:
        """基于模板生成患者回复
        
        Args:
            model: 动态认知模型
            therapist_message: 治疗师消息
            
        Returns:
            生成的患者回复
        """
        # 确定当前主要情绪类型
        emotion_type = self._determine_dominant_emotion(model.current_state)
        
        # 选择对应情绪的回复模板
        templates = self.response_templates.get(emotion_type, self.response_templates["neutral"])
        template = random.choice(templates)
        
        # 选择一个活跃的核心信念
        belief = self._get_active_belief(model.current_state)
        
        # 填充模板
        response = template.format(belief=belief)
        
        # 将回复添加到对话历史
        model.add_message_to_history("patient", response)
        
        return response
    
    def _determine_dominant_emotion(self, state: StateSnapshot) -> str:
        """确定主导情绪类型
        
        Args:
            state: 当前状态快照
            
        Returns:
            情绪类别名称
        """
        if not state.emotions:
            return "neutral"
        
        # 按强度排序情绪
        sorted_emotions = sorted(
            state.emotions.items(), 
            key=lambda x: x[1].intensity, 
            reverse=True
        )
        
        # 获取最强情绪
        dominant_emotion = sorted_emotions[0][1].type.lower()
        
        # 映射到情绪类别
        for keyword, category in self.emotion_mapping.items():
            if keyword in dominant_emotion:
                return category
        
        return "neutral"
    
    def _get_active_belief(self, state: StateSnapshot) -> str:
        """获取一个活跃的核心信念
        
        Args:
            state: 当前状态快照
            
        Returns:
            核心信念内容
        """
        # 尝试获取激活度较高的核心信念
        active_beliefs = [
            belief for _, belief in state.core_beliefs.items() 
            if belief.activation_level > 0.6
        ]
        
        if active_beliefs:
            return active_beliefs[0].content
        
        # 如果没有高激活度的核心信念，返回一个自动思维
        if state.automatic_thoughts:
            auto_thought = next(iter(state.automatic_thoughts.values()))
            return auto_thought.content
        
        # 如果都没有，返回一个默认陈述
        return "我不确定我在想什么..."