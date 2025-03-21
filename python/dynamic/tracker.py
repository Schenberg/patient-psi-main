import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
from pydantic import BaseModel

# u5bfcu5165u52a8u6001u8ba4u77e5u6a21u578b
from .cognitive_model import DynamicCognitiveModel, StateSnapshot

class ChangeMetric(BaseModel):
    """u5b9au4e49u4e00u4e2au53d8u5316u6307u6807"""
    name: str
    description: str
    values: List[float] = []  # u5b58u50a8u6307u6807u503cu5386u53f2
    timestamps: List[str] = []  # u5bf9u5e94u7684u65f6u95f4u6233
    
    def add_value(self, value: float, timestamp: str):
        """u6dfbu52a0u65b0u7684u6307u6807u503cu548cu65f6u95f4u6233"""
        self.values.append(value)
        self.timestamps.append(timestamp)
    
    def get_trend(self, window: int = 3) -> float:
        """u8ba1u7b97u6700u8fd1windowu4e2au6570u636eu70b9u7684u53d8u5316u8d8bu52bf
        
        Args:
            window: u7a97u53e3u5927u5c0fuff0cu9ed8u8ba4u4e3a3
            
        Returns:
            u53d8u5316u8d8bu52bfu7cfbu6570uff0cu6b63u503cu8868u793au6307u6807u4e0au5347uff0cu8d1fu503cu8868u793au6307u6807u4e0bu964d
        """
        if len(self.values) < window:
            return 0.0
            
        recent_values = self.values[-window:]
        # u7b80u5355u7ebfu6027u56deu5f52
        x = np.arange(window)
        y = np.array(recent_values)
        slope = np.polyfit(x, y, 1)[0]
        return slope

class TherapyProgressTracker:
    """u6cbbu7597u8fdbu5c55u8ddfu8e2au5668uff0cu76d1u63a7u5e76u5206u6790u60a3u8005u7684u5fc3u7406u72b6u6001u53d8u5316"""
    
    def __init__(self, output_dir: str = "./reports"):
        """u521du59cbu5316u8ddfu8e2au5668
        
        Args:
            output_dir: u62a5u544au8f93u51fau76eeu5f55
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # u521du59cbu5316u6307u6807
        self.metrics: Dict[str, ChangeMetric] = {
            "depression_level": ChangeMetric(
                name="depression_level",
                description="u6291u90c1u6c34u5e73"
            ),
            "belief_flexibility": ChangeMetric(
                name="belief_flexibility",
                description="u4fe1u5ff5u7075u6d3bu6027"
            ),
            "emotion_regulation": ChangeMetric(
                name="emotion_regulation",
                description="u60c5u7eeau8c03u8282u80fdu529b"
            ),
            "coping_effectiveness": ChangeMetric(
                name="coping_effectiveness",
                description="u5e94u5bf9u7b56u7565u6709u6548u6027"
            )
        }
    
    def track_state_change(self, model: DynamicCognitiveModel):
        """u8ddfu8e2au6a21u578bu72b6u6001u53d8u5316uff0cu66f4u65b0u6307u6807
        
        Args:
            model: u52a8u6001u8ba4u77e5u6a21u578b
        """
        # u83b7u53d6u5f53u524du72b6u6001
        current_state = model.current_state
        timestamp = current_state.timestamp
        
        # u66f4u65b0u6291u90c1u6c34u5e73u6307u6807
        self.metrics["depression_level"].add_value(
            value=current_state.depression_level,
            timestamp=timestamp
        )
        
        # u8ba1u7b97u4fe1u5ff5u7075u6d3bu6027u6307u6807 (u57fau4e8eu6838u5fc3u4fe1u5ff5u7684u6fc0u6d3bu5ea6)
        core_belief_activation = [b.activation_level for b in current_state.core_beliefs.values()]
        avg_activation = sum(core_belief_activation) / len(core_belief_activation) if core_belief_activation else 0.5
        belief_flexibility = 1.0 - avg_activation  # u6fc0u6d3bu5ea6u8d8au4f4euff0cu7075u6d3bu6027u8d8au9ad8
        
        self.metrics["belief_flexibility"].add_value(
            value=belief_flexibility,
            timestamp=timestamp
        )
        
        # u8ba1u7b97u60c5u7eeau8c03u8282u6307u6807
        # u8ba1u7b97u6d88u6781u60c5u7eeau5360u6bd4
        negative_emotions = [
            "sad", "depressed", "anxious", "angry", "frustrated", 
            "hopeless", "guilty", "ashamed", "worthless", "lonely"
        ]
        
        negative_count = 0
        total_count = len(current_state.emotions)
        
        if total_count > 0:
            for emotion in current_state.emotions.values():
                if any(neg in emotion.type.lower() for neg in negative_emotions):
                    negative_count += 1
            
            negative_ratio = negative_count / total_count
            emotion_regulation = 1.0 - negative_ratio
        else:
            emotion_regulation = 0.5  # u9ed8u8ba4u503c
        
        self.metrics["emotion_regulation"].add_value(
            value=emotion_regulation,
            timestamp=timestamp
        )
        
        # u8ba1u7b97u5e94u5bf9u7b56u7565u6709u6548u6027
        coping_scores = [s.effectiveness for s in current_state.coping_strategies.values()]
        avg_coping = sum(coping_scores) / len(coping_scores) if coping_scores else 0.5
        
        self.metrics["coping_effectiveness"].add_value(
            value=avg_coping,
            timestamp=timestamp
        )
    
    def analyze_progress(self, model: DynamicCognitiveModel, window: int = 3) -> Dict[str, Any]:
        """u5206u6790u6cbbu7597u8fdbu5c55
        
        Args:
            model: u52a8u6001u8ba4u77e5u6a21u578b
            window: u8d8bu52bfu5206u6790u7a97u53e3u5927u5c0f
            
        Returns:
            u6cbbu7597u8fdbu5c55u5206u6790u7ed3u679c
        """
        # u5148u66f4u65b0u6307u6807
        self.track_state_change(model)
        
        # u8ba1u7b97u5404u6307u6807u8d8bu52bf
        trends = {}
        for metric_name, metric in self.metrics.items():
            trends[metric_name] = {
                "trend": metric.get_trend(window),
                "current": metric.values[-1] if metric.values else 0.0,
                "previous": metric.values[-2] if len(metric.values) > 1 else 0.0
            }
        
        # u6839u636eu6307u6807u53d8u5316u751fu6210u5efau8bae
        suggestions = {}
        
        # u6291u90c1u6c34u5e73u5efau8bae
        depression_trend = trends["depression_level"]["trend"]
        if depression_trend > 0.05:  # u6291u90c1u6c34u5e73u4e0au5347
            suggestions["depression"] = [
                "u60a3u8005u7684u6291u90c1u6c34u5e73u6709u6240u4e0au5347uff0cu53efu80fdu9700u8981u52a0u5f3au6df1u5165u63a2u8ba8u6291u90c1u89e6u53d1u56e0u7d20",
                "u5efau8baeu5f15u5bfcu60a3u8005u8bc6u522bu548cu8d28u7591u6d88u6781u81eau52a8u601du7ef4"
            ]
        elif depression_trend < -0.05:  # u6291u90c1u6c34u5e73u4e0bu964d
            suggestions["depression"] = [
                "u60a3u8005u7684u6291u90c1u6c34u5e73u6709u6240u4e0bu964duff0cu53efu4ee5u5f3au5316u8fd9u4e00u79efu6781u53d8u5316",
                "u9f13u52b1u60a3u8005u603bu7ed3u5e2eu52a9u81eau5df1u7684u7b56u7565u548cu8fdbu6b65"
            ]
        
        # u4fe1u5ff5u7075u6d3bu6027u5efau8bae
        flexibility_trend = trends["belief_flexibility"]["trend"]
        if flexibility_trend < -0.05:  # u4fe1u5ff5u7075u6d3bu6027u4e0bu964d
            suggestions["belief"] = [
                "u60a3u8005u7684u4fe1u5ff5u53d8u5f97u66f4u50f5u5316uff0cu53efu4ee5u5c1du8bd5u4f7fu7528u8ba4u77e5u91cdu6784u6280u672f",
                "u63a2u7d22u8ba4u77e5u6241u66f2u6a21u5f0fu5e76u9f13u52b1u591au89d2u5ea6u601du8003"
            ]
        
        # u60c5u7eeau8c03u8282u5efau8bae
        emotion_trend = trends["emotion_regulation"]["trend"]
        if emotion_trend < -0.05:  # u60c5u7eeau8c03u8282u80fdu529bu4e0bu964d
            suggestions["emotion"] = [
                "u60a3u8005u7684u60c5u7eeau8c03u8282u80fdu529bu6709u6240u4e0bu964duff0cu53efu4ee5u7ed9u4e88u66f4u591au60c5u7eeau8c03u8282u6280u5de7",
                "u5f15u5bfcu60c5u7eeau8bc6u522bu548cu63a5u7eb3u7ec3u4e60"
            ]
        
        return {
            "trends": trends,
            "suggestions": suggestions,
            "session_count": len(model.state_history) + 1
        }
    
    def generate_report(self, model: DynamicCognitiveModel) -> str:
        """u751fu6210u6cbbu7597u8fdbu5c55u62a5u544a
        
        Args:
            model: u52a8u6001u8ba4u77e5u6a21u578b
            
        Returns:
            u62a5u544au6587u4ef6u8defu5f84
        """
        # u5206u6790u8fdbu5c55
        progress = self.analyze_progress(model)
        
        # u521bu5efau62a5u544au76eeu5f55
        report_dir = os.path.join(self.output_dir, model.patient_id)
        os.makedirs(report_dir, exist_ok=True)
        
        # u751fu6210u56feu8868
        self._generate_trend_chart(model, report_dir)
        
        # u751fu6210u62a5u544au6587u4ef6
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(report_dir, f"progress_report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        
        return report_file
    
    def _generate_trend_chart(self, model: DynamicCognitiveModel, output_dir: str) -> str:
        """u751fu6210u8d8bu52bfu56feu8868
        
        Args:
            model: u52a8u6001u8ba4u77e5u6a21u578b
            output_dir: u8f93u51fau76eeu5f55
            
        Returns:
            u56feu8868u6587u4ef6u8defu5f84
        """
        # u8bbeu7f6eu56feu8868
        plt.figure(figsize=(12, 8))
        
        # u9009u53d6u8981u663eu793au7684u6307u6807
        display_metrics = ["depression_level", "belief_flexibility", "emotion_regulation", "coping_effectiveness"]
        
        for i, metric_name in enumerate(display_metrics):
            metric = self.metrics[metric_name]
            if not metric.values:
                continue
                
            plt.subplot(2, 2, i+1)
            plt.plot(range(len(metric.values)), metric.values, 'o-')
            
            # u6dfbu52a0u8d8bu52bfu7ebf
            if len(metric.values) > 2:
                x = np.arange(len(metric.values))
                z = np.polyfit(x, metric.values, 1)
                p = np.poly1d(z)
                plt.plot(x, p(x), "r--", alpha=0.7)
            
            plt.title(metric.description)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            if len(metric.values) > 5:
                # u4ec5u663eu793au90e8u5206u65f6u95f4u70b9u6807u7b7e
                indices = np.linspace(0, len(metric.timestamps)-1, 5).astype(int)
                plt.xticks(indices, [metric.timestamps[i].split('T')[0] for i in indices], rotation=45)
            else:
                plt.xticks(range(len(metric.timestamps)), [ts.split('T')[0] for ts in metric.timestamps], rotation=45)
        
        plt.tight_layout()
        
        # u4fddu5b58u56feu8868
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_file = os.path.join(output_dir, f"progress_chart_{timestamp}.png")
        plt.savefig(chart_file)
        plt.close()
        
        return chart_file