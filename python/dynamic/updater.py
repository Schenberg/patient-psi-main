import os
import json
from typing import Dict, List, Any, Optional, Tuple
import openai
from pydantic import BaseModel
from datetime import datetime

# 导入动态认知模型
from .cognitive_model import DynamicCognitiveModel, StateSnapshot

class UpdatePrompt(BaseModel):
    """用于生成更新提示的模板"""
    system_prompt: str
    user_prompt_template: str
    
    @classmethod
    def default(cls) -> "UpdatePrompt":
        """创建默认的更新提示模板"""
        system_prompt = """
        你是一位专业的认知行为治疗分析师，负责评估治疗会话中患者的认知和情绪变化。
        基于治疗师的干预，你需要分析患者的认知状态如何变化，包括核心信念、中间信念、自动思维、情绪、行为和应对策略的变化。
        请根据治疗师的消息给出具体、合理的变化估计。
        """
        
        user_prompt_template = """
        基于以下患者信息和治疗师的消息，分析可能发生的认知和情绪变化：
        
        患者基本信息：
        - 姓名: {name}
        - 历史背景: {history}
        
        患者当前认知状态：
        1. 核心信念: 
        {core_beliefs}
        
        2. 中间信念: 
        {intermediate_beliefs}
        
        3. 自动思维: 
        {automatic_thoughts}
        
        4. 情绪: 
        {emotions}
        
        5. 行为: 
        {behaviors}
        
        6. 应对策略: 
        {coping_strategies}
        
        7. 抑郁水平: {depression_level}
        
        最近的对话历史：
        {conversation_history}
        
        治疗师最新消息: 
        "{therapist_message}"
        
        请分析这条消息可能对患者认知状态产生的影响，并以JSON格式返回可能的变化（仅返回有变化的部分）。
        例如:
        ```json
        {
          "depression_level": 0.6,
          "core_beliefs": {
            "helpless_0": {"activation_level": 0.4},
            "new_belief": {"content": "我可以解决一些问题", "type": "helpless", "strength": 0.3, "activation_level": 0.6}
          },
          "emotions": {
            "emotion_0": {"intensity": 0.7},
            "new_emotion": {"type": "希望", "intensity": 0.4}
          }
        }
        ```
        
        你的完整分析（仅包含JSON响应，不要有其他文字）：
        """
        
        return cls(system_prompt=system_prompt, user_prompt_template=user_prompt_template)

class CognitiveUpdateEngine:
    """认知更新引擎，分析治疗师消息并计算状态更新"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """初始化更新引擎
        
        Args:
            api_key: OpenAI API密钥，如果为None则尝试从环境变量获取
            model: 使用的OpenAI模型名称
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API密钥未提供，请通过参数传入或设置OPENAI_API_KEY环境变量")
        
        self.model = model
        self.update_prompt = UpdatePrompt.default()
        openai.api_key = self.api_key
    
    def compute_updates(self, model: DynamicCognitiveModel, therapist_message: str) -> Dict[str, Any]:
        """计算基于治疗师消息的认知状态更新
        
        Args:
            model: 当前的动态认知模型
            therapist_message: 治疗师的最新消息
            
        Returns:
            包含各种认知组件更新的字典
        """
        # 准备模型状态信息
        prompt_data = self._prepare_prompt_data(model, therapist_message)
        
        # 尝试基于认知链选择适当的反应
        cognitive_chain = self._select_cognitive_chain(model, therapist_message)
        
        # 如果找到合适的认知链，将其整合到更新中
        if cognitive_chain:
            updates = self._generate_updates_with_cognitive_chain(model, therapist_message, cognitive_chain)
        else:
            # 调用LLM生成更新（原有逻辑）
            updates = self._generate_updates_with_llm(prompt_data)
        
        return updates

    def _select_cognitive_chain(self, model: DynamicCognitiveModel, therapist_message: str) -> Optional[Tuple[str, Any]]:
        """基于治疗师消息选择合适的认知链
        
        Args:
            model: 当前的动态认知模型
            therapist_message: 治疗师的最新消息
            
        Returns:
            匹配的认知链ID和认知链对象的元组，如果未找到则返回None
        """
        # 简单关键词匹配策略
        # 在实际应用中，可以使用更复杂的语义匹配或机器学习方法
        for chain_id, chain in model.current_state.cognitive_chains.items():
            # 检查情境是否匹配治疗师消息
            if chain.situation.lower() in therapist_message.lower():
                # 找到匹配的认知链
                return (chain_id, chain)
                
            # 检查线索是否匹配
            if chain.clue and chain.clue.lower() in therapist_message.lower():
                return (chain_id, chain)
        
        # 未找到匹配的认知链
        return None
    
    def _generate_updates_with_cognitive_chain(self, model: DynamicCognitiveModel, therapist_message: str, 
                                              chain_info: Tuple[str, Any]) -> Dict[str, Any]:
        """基于选择的认知链生成状态更新
        
        Args:
            model: 当前的动态认知模型
            therapist_message: 治疗师的最新消息
            chain_info: 包含认知链ID和对象的元组
            
        Returns:
            包含各种认知组件更新的字典
        """
        chain_id, chain = chain_info
        
        # 创建基于认知链的更新
        updates = {}
        
        # 1. 更新认知链激活程度
        updates['cognitive_chains'] = {
            chain_id: {
                'activation_level': min(chain.activation_level + 0.2, 1.0)  # 增加激活程度
            }
        }
        
        # 2. 基于认知链的思维更新自动思维
        # 假设我们要添加一个新的自动思维，基于认知链的thought
        thought_id = f"thought_from_chain_{chain_id}"
        updates['automatic_thoughts'] = {
            thought_id: {
                'content': chain.thought,
                'strength': 0.7,
                'situation': therapist_message[:50]  # 使用治疗师消息作为情境
            }
        }
        
        # 3. 基于认知链的情绪更新情绪强度
        # 假设我们在emotions中查找匹配的情绪类型
        emotion_updated = False
        if 'emotions' not in updates:
            updates['emotions'] = {}
            
        for emotion_id, emotion in model.current_state.emotions.items():
            if chain.emotion.lower() in emotion.type.lower():
                # 找到匹配的情绪，更新其强度
                intensity_change = 0.1 * chain.polarity  # 根据极性决定增强或减弱
                updates['emotions'][emotion_id] = {
                    'intensity': max(0.0, min(1.0, emotion.intensity + intensity_change))
                }
                emotion_updated = True
                break
                
        # 如果未找到匹配的情绪，创建新情绪
        if not emotion_updated:
            new_emotion_id = f"emotion_from_chain_{chain_id}"
            updates['emotions'][new_emotion_id] = {
                'type': chain.emotion,
                'intensity': 0.6 if chain.polarity > 0 else 0.7  # 消极情绪初始强度更高
            }
        
        # 4. 基于认知链的行为更新行为频率
        # 类似情绪的处理方式
        behavior_updated = False
        if 'behaviors' not in updates:
            updates['behaviors'] = {}
            
        for behavior_id, behavior in model.current_state.behaviors.items():
            if chain.action.lower() in behavior.description.lower():
                # 找到匹配的行为，更新其频率
                updates['behaviors'][behavior_id] = {
                    'frequency': max(0.0, min(1.0, behavior.frequency + 0.1))
                }
                behavior_updated = True
                break
                
        # 如果未找到匹配的行为，创建新行为
        if not behavior_updated:
            new_behavior_id = f"behavior_from_chain_{chain_id}"
            updates['behaviors'][new_behavior_id] = {
                'description': chain.action,
                'frequency': 0.6
            }
        
        # 5. 根据认知链的极性更新抑郁水平
        # 消极极性增加抑郁水平，积极极性降低抑郁水平
        depression_change = -0.05 * chain.polarity  # 反向关系
        updates['depression_level'] = max(0.0, min(1.0, model.current_state.depression_level + depression_change))
        
        return updates
    
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
        
        # 格式化核心信念
        core_beliefs_text = "\n".join([f"- {id}: {belief.content} (类型: {belief.type}, 强度: {belief.strength}, 激活度: {belief.activation_level})" 
                              for id, belief in state.core_beliefs.items()])
        
        # 格式化中间信念
        intermediate_beliefs_text = "\n".join([f"- {id}: {belief.content} (类型: {belief.type}, 强度: {belief.strength}, 激活度: {belief.activation_level})" 
                                    for id, belief in state.intermediate_beliefs.items()])
        
        # 格式化自动思维
        automatic_thoughts_text = "\n".join([f"- {id}: {thought.content} (强度: {thought.strength}, 情境: {thought.situation or '未指定'})" 
                                for id, thought in state.automatic_thoughts.items()])
        
        # 格式化情绪
        emotions_text = "\n".join([f"- {id}: {emotion.type} (强度: {emotion.intensity})" 
                       for id, emotion in state.emotions.items()])
        
        # 格式化行为
        behaviors_text = "\n".join([f"- {id}: {behavior.description} (频率: {behavior.frequency})" 
                         for id, behavior in state.behaviors.items()])
        
        # 格式化应对策略
        coping_strategies_text = "\n".join([f"- {id}: {strategy.description} (有效性: {strategy.effectiveness}, 使用频率: {strategy.usage_frequency})" 
                                for id, strategy in state.coping_strategies.items()])
        
        # 格式化对话历史
        conversation_history = model.get_recent_conversation(5)
        conversation_history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        
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
    
    def _generate_updates_with_llm(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用LLM生成认知更新
        
        Args:
            prompt_data: 模板填充数据
            
        Returns:
            包含各种认知组件更新的字典
        """
        try:
            # 填充用户提示模板
            user_prompt = self.update_prompt.user_prompt_template.format(**prompt_data)
            
            # 准备消息
            messages = [
                {"role": "system", "content": self.update_prompt.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 调用OpenAI API
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # 提取并解析结果
            result = response.choices[0].message.content.strip()
            
            # 清理结果，移除可能的Markdown代码块标记
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            # 进一步清理和修复 JSON 格式问题
            result = result.strip()
            # 确保开始有花括号
            if not result.startswith("{"):
                if '"' in result and ':' in result:
                    # 尝试修复缺少花括号的情况
                    result = "{" + result
            # 确保结束有花括号
            if not result.endswith("}"):
                if result.count('{') > result.count('}'):
                    result = result + "}"
            
            try:
                # 解析JSON
                updates = json.loads(result.strip())
                return updates
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {str(e)}，尝试进行更强的修复")
                try:
                    # 尝试更强的修复方式 - 提取键值对
                    import re
                    pattern = r'"([^"]+)"\s*:\s*([^,\}]+)'
                    matches = re.findall(pattern, result)
                    if matches:
                        # 手动构建字典
                        fixed_dict = {}
                        for key, value in matches:
                            # 尝试将值转换为适当的类型
                            try:
                                if value.strip().startswith('"') and value.strip().endswith('"'):
                                    # 字符串值
                                    fixed_dict[key] = value.strip('" \t\n\r')
                                elif value.strip().lower() in ['true', 'false']:
                                    # 布尔值
                                    fixed_dict[key] = value.strip().lower() == 'true'
                                else:
                                    # 尝试作为数字
                                    try:
                                        fixed_dict[key] = float(value.strip())
                                    except:
                                        fixed_dict[key] = value.strip()
                            except:
                                fixed_dict[key] = value.strip()
                        return fixed_dict
                    return {}
                except Exception as inner_e:
                    print(f"增强修复也失败: {str(inner_e)}")
                    return {}
        except Exception as e:
            print(f"生成更新时发生错误: {str(e)}")
            # 返回空更新
            return {}
    
    def update_model(self, model: DynamicCognitiveModel, therapist_message: str) -> Tuple[DynamicCognitiveModel, Dict[str, Any]]:
        """更新认知模型
        
        Args:
            model: 当前的动态认知模型
            therapist_message: 治疗师的消息
            
        Returns:
            更新后的模型和应用的更新内容
        """
        # 添加治疗师消息到对话历史
        model.add_message_to_history("therapist", therapist_message)
        
        # 计算更新
        updates = self.compute_updates(model, therapist_message)
        
        # 应用更新
        if updates:
            model.update_state(updates, f"therapist_message:{datetime.now().isoformat()}")
            
        return model, updates


class UpdateWeights(BaseModel):
    """更新权重配置"""
    helpless_belief: float = 0.3
    unlovable_belief: float = 0.3
    worthless_belief: float = 0.3
    intermediate_belief: float = 0.25
    automatic_thought: float = 0.4
    emotion: float = 0.6
    behavior: float = 0.2
    coping_strategy: float = 0.15
    depression_level: float = 0.3
    
    @classmethod
    def default(cls) -> "UpdateWeights":
        """默认更新权重"""
        return cls()


class SimpleUpdateEngine:
    """简单更新引擎，基于规则和关键词分析更新认知状态"""
    
    def __init__(self, weights: Optional[UpdateWeights] = None):
        """初始化简单更新引擎
        
        Args:
            weights: 更新权重配置
        """
        self.weights = weights or UpdateWeights.default()
        
        # 定义积极/消极关键词
        self.positive_keywords = [
            "希望", "进步", "改善", "积极", "成长", "能够", "可以", "强项", "有价值", "有意义", 
            "解决", "应对", "成功", "成就", "价值", "优势", "学习", "理解", "接纳", "支持"
        ]
        
        self.negative_keywords = [
            "困难", "问题", "挣扎", "失败", "无望", "无助", "无价值", "负担", "悲伤", "抑郁", 
            "焦虑", "痛苦", "孤独", "绝望", "自责", "批判", "无法", "不能", "永远不会", "没人"
        ]
    
    def compute_updates(self, model: DynamicCognitiveModel, therapist_message: str) -> Dict[str, Any]:
        """基于关键词和简单规则计算更新
        
        Args:
            model: 当前的动态认知模型
            therapist_message: 治疗师的消息
            
        Returns:
            包含各种认知组件更新的字典
        """
        # 分析消息的情感倾向
        sentiment_score = self._analyze_sentiment(therapist_message)
        
        # 初始化更新字典
        updates = {}
        
        # 根据情感得分调整抑郁水平
        if sentiment_score != 0:
            updates["depression_level"] = max(0.0, min(1.0, 
                model.current_state.depression_level - sentiment_score * self.weights.depression_level))
        
        # 调整核心信念激活水平
        updates["core_beliefs"] = {}
        for belief_id, belief in model.current_state.core_beliefs.items():
            # 针对不同类型的信念分别调整
            if belief.type == "helpless" and sentiment_score != 0:
                activation_adjustment = -sentiment_score * self.weights.helpless_belief
                updates["core_beliefs"][belief_id] = {
                    "activation_level": max(0.1, min(1.0, belief.activation_level + activation_adjustment))
                }
            elif belief.type == "unlovable" and sentiment_score != 0:
                activation_adjustment = -sentiment_score * self.weights.unlovable_belief
                updates["core_beliefs"][belief_id] = {
                    "activation_level": max(0.1, min(1.0, belief.activation_level + activation_adjustment))
                }
            elif belief.type == "worthless" and sentiment_score != 0:
                activation_adjustment = -sentiment_score * self.weights.worthless_belief
                updates["core_beliefs"][belief_id] = {
                    "activation_level": max(0.1, min(1.0, belief.activation_level + activation_adjustment))
                }
        
        # 如果没有任何核心信念更新，删除空字典
        if not updates["core_beliefs"]:
            del updates["core_beliefs"]
            
        # 类似地处理其他认知组件的更新...
        # 这里只是简化示例，实际应用中可以添加更多复杂的更新逻辑
            
        return updates
    
    def _analyze_sentiment(self, message: str) -> float:
        """分析消息的情感得分
        
        Args:
            message: 要分析的消息
            
        Returns:
            情感得分，正值表示积极，负值表示消极，范围[-1, 1]
        """
        positive_count = sum(1 for word in self.positive_keywords if word in message)
        negative_count = sum(1 for word in self.negative_keywords if word in message)
        
        total_count = positive_count + negative_count
        if total_count == 0:
            return 0
            
        return (positive_count - negative_count) / total_count