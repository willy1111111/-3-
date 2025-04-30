"""
百家乐AI系统自学习模块
支持模型评估、策略优化和自适应学习
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Learner")

class ModelType(Enum):
    """模型类型"""
    DEEPSEEK = "deepseek"
    LLAMA = "llama"
    MISTRAL = "mistral"
    CNN_RNN = "cnn_rnn"

class StrategyType(Enum):
    """策略类型"""
    FOLLOW = "follow"  # 跟随预测
    REVERSE = "reverse"  # 反向预测
    TREND = "trend"  # 趋势跟踪
    PATTERN = "pattern"  # 模式识别

@dataclass
class LearningConfig:
    """学习配置"""
    enabled: bool = True
    batch_size: int = 32
    learning_interval: int = 3600  # 学习间隔(秒)
    min_samples: int = 1000  # 最小样本数
    performance_threshold: float = 0.6  # 性能阈值
    max_models: int = 5  # 最大模型数量
    max_strategies: int = 3  # 最大策略数量

@dataclass
class ModelPerformance:
    """模型性能"""
    model_id: str
    model_type: ModelType
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    predictions: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    def _asdict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "model_type": self.model_type.value,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "predictions": self.predictions,
            "last_update": self.last_update.isoformat()
        }

@dataclass
class StrategyPerformance:
    """策略性能"""
    strategy_id: str
    strategy_type: StrategyType
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trades: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    def _asdict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "strategy_id": self.strategy_id,
            "strategy_type": self.strategy_type.value,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "trades": self.trades,
            "last_update": self.last_update.isoformat()
        }

class Learner:
    """自学习器"""
    
    def __init__(self, config: LearningConfig):
        """
        初始化自学习器
        
        Args:
            config: 学习配置
        """
        self.config = config
        self.running = False
        self.last_learn_time = datetime.now()
        
        # 性能跟踪
        self.model_performances: Dict[str, ModelPerformance] = {}
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        
        # 学习数据缓冲
        self.prediction_buffer = []
        self.trade_buffer = []
    
    async def start(self):
        """启动自学习器"""
        if not self.config.enabled:
            return
            
        self.running = True
        asyncio.create_task(self._learning_loop())
        logger.info("自学习器已启动")
    
    async def stop(self):
        """停止自学习器"""
        self.running = False
        logger.info("自学习器已停止")
    
    async def _learning_loop(self):
        """学习循环"""
        while self.running:
            try:
                # 检查是否需要学习
                if (datetime.now() - self.last_learn_time).total_seconds() >= self.config.learning_interval:
                    await self._perform_learning()
                    self.last_learn_time = datetime.now()
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"学习循环出错: {str(e)}")
                await asyncio.sleep(5)
    
    async def _perform_learning(self):
        """执行学习"""
        try:
            # 检查样本数量
            if len(self.prediction_buffer) < self.config.min_samples:
                logger.info(f"样本数量不足: {len(self.prediction_buffer)} < {self.config.min_samples}")
                return
            
            # 更新模型性能
            await self._update_model_performance()
            
            # 更新策略性能
            await self._update_strategy_performance()
            
            # 优化模型组合
            await self._optimize_model_ensemble()
            
            # 优化策略组合
            await self._optimize_strategy_ensemble()
            
            # 清理缓冲区
            self._clear_buffers()
            
            logger.info("学习完成")
            
        except Exception as e:
            logger.error(f"学习过程出错: {str(e)}")
    
    async def _update_model_performance(self):
        """更新模型性能"""
        try:
            # 按模型分组预测结果
            model_predictions = {}
            for record in self.prediction_buffer:
                model_id = record["model_id"]
                if model_id not in model_predictions:
                    model_predictions[model_id] = {
                        "y_true": [],
                        "y_pred": [],
                        "type": record["model_type"]
                    }
                
                model_predictions[model_id]["y_true"].append(record["actual"])
                model_predictions[model_id]["y_pred"].append(record["prediction"])
            
            # 计算每个模型的性能指标
            for model_id, data in model_predictions.items():
                y_true = np.array(data["y_true"])
                y_pred = np.array(data["y_pred"])
                
                performance = ModelPerformance(
                    model_id=model_id,
                    model_type=data["type"],
                    accuracy=accuracy_score(y_true, y_pred),
                    precision=precision_score(y_true, y_pred, average="weighted"),
                    recall=recall_score(y_true, y_pred, average="weighted"),
                    f1=f1_score(y_true, y_pred, average="weighted"),
                    predictions=len(y_true),
                    last_update=datetime.now()
                )
                
                self.model_performances[model_id] = performance
                
            logger.info(f"已更新{len(model_predictions)}个模型的性能指标")
            
        except Exception as e:
            logger.error(f"更新模型性能出错: {str(e)}")
    
    async def _update_strategy_performance(self):
        """更新策略性能"""
        try:
            # 按策略分组交易结果
            strategy_trades = {}
            for record in self.trade_buffer:
                strategy_id = record["strategy_id"]
                if strategy_id not in strategy_trades:
                    strategy_trades[strategy_id] = {
                        "wins": 0,
                        "losses": 0,
                        "profit": 0.0,
                        "loss": 0.0,
                        "type": record["strategy_type"]
                    }
                
                if record["profit"] > 0:
                    strategy_trades[strategy_id]["wins"] += 1
                    strategy_trades[strategy_id]["profit"] += record["profit"]
                else:
                    strategy_trades[strategy_id]["losses"] += 1
                    strategy_trades[strategy_id]["loss"] += abs(record["profit"])
            
            # 计算每个策略的性能指标
            for strategy_id, data in strategy_trades.items():
                total_trades = data["wins"] + data["losses"]
                win_rate = data["wins"] / total_trades if total_trades > 0 else 0
                profit_factor = data["profit"] / data["loss"] if data["loss"] > 0 else float("inf")
                
                performance = StrategyPerformance(
                    strategy_id=strategy_id,
                    strategy_type=data["type"],
                    win_rate=win_rate,
                    profit_factor=profit_factor,
                    trades=total_trades,
                    last_update=datetime.now()
                )
                
                self.strategy_performances[strategy_id] = performance
                
            logger.info(f"已更新{len(strategy_trades)}个策略的性能指标")
            
        except Exception as e:
            logger.error(f"更新策略性能出错: {str(e)}")
    
    async def _optimize_model_ensemble(self):
        """优化模型组合"""
        try:
            # 按性能排序模型
            sorted_models = sorted(
                self.model_performances.values(),
                key=lambda x: (x.accuracy, x.f1),
                reverse=True
            )
            
            # 保留性能最好的N个模型
            top_models = sorted_models[:self.config.max_models]
            
            # 移除性能不达标的模型
            for model in sorted_models[self.config.max_models:]:
                if model.model_id in self.model_performances:
                    del self.model_performances[model.model_id]
            
            logger.info(f"模型组合优化完成，保留{len(top_models)}个模型")
            
        except Exception as e:
            logger.error(f"优化模型组合出错: {str(e)}")
    
    async def _optimize_strategy_ensemble(self):
        """优化策略组合"""
        try:
            # 按性能排序策略
            sorted_strategies = sorted(
                self.strategy_performances.values(),
                key=lambda x: (x.win_rate, x.profit_factor),
                reverse=True
            )
            
            # 保留性能最好的N个策略
            top_strategies = sorted_strategies[:self.config.max_strategies]
            
            # 移除性能不达标的策略
            for strategy in sorted_strategies[self.config.max_strategies:]:
                if strategy.strategy_id in self.strategy_performances:
                    del self.strategy_performances[strategy.strategy_id]
            
            logger.info(f"策略组合优化完成，保留{len(top_strategies)}个策略")
            
        except Exception as e:
            logger.error(f"优化策略组合出错: {str(e)}")
    
    def _clear_buffers(self):
        """清理数据缓冲区"""
        self.prediction_buffer = []
        self.trade_buffer = []
    
    def add_prediction_result(self, result: Dict[str, Any]):
        """添加预测结果"""
        # 复制结果并添加时间戳
        prediction_record = result.copy()
        if "timestamp" not in prediction_record:
            prediction_record["timestamp"] = datetime.now().isoformat()
        self.prediction_buffer.append(prediction_record)
    
    def add_trade_result(self, result: Dict[str, Any]):
        """添加交易结果"""
        # 复制结果并添加时间戳
        trade_record = result.copy()
        if "timestamp" not in trade_record:
            trade_record["timestamp"] = datetime.now().isoformat()
        self.trade_buffer.append(trade_record)
    
    def get_best_model(self) -> Optional[Tuple[str, ModelPerformance]]:
        """
        获取最佳模型
        
        Returns:
            Optional[Tuple[str, ModelPerformance]]: 模型ID和性能
        """
        if not self.model_performances:
            return None
            
        best_model_id = max(
            self.model_performances.items(),
            key=lambda x: (x[1].accuracy, x[1].f1)
        )[0]
        
        return best_model_id, self.model_performances[best_model_id]
    
    def get_best_strategy(self) -> Optional[Tuple[str, StrategyPerformance]]:
        """
        获取最佳策略
        
        Returns:
            Optional[Tuple[str, StrategyPerformance]]: 策略ID和性能
        """
        if not self.strategy_performances:
            return None
            
        best_strategy_id = max(
            self.strategy_performances.items(),
            key=lambda x: (x[1].win_rate, x[1].profit_factor)
        )[0]
        
        return best_strategy_id, self.strategy_performances[best_strategy_id]
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        获取学习统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        # 获取最佳模型和策略
        best_model = self.get_best_model()
        best_strategy = self.get_best_strategy()
        
        # 构建统计信息
        stats = {
            "models_count": len(self.model_performances),
            "strategies_count": len(self.strategy_performances),
            "prediction_samples": len(self.prediction_buffer),
            "trade_samples": len(self.trade_buffer),
            "best_model": best_model[1]._asdict() if best_model else None,
            "best_strategy": best_strategy[1]._asdict() if best_strategy else None,
            "last_learn_time": self.last_learn_time.isoformat(),
            "learning_enabled": self.config.enabled
        }
        
        return stats 