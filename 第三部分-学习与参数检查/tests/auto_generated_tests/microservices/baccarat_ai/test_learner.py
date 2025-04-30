import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import numpy as np

from microservices.baccarat_ai.learner import (
    Learner,
    LearningConfig,
    ModelType,
    StrategyType,
    ModelPerformance,
    StrategyPerformance
)

class TestLearner:
    """测试百家乐AI学习器功能"""
    
    @pytest.fixture
    def learning_config(self):
        """返回学习配置的fixture"""
        return LearningConfig(
            enabled=True,
            batch_size=32,
            learning_interval=3600,
            min_samples=10,  # 测试使用更小的值
            performance_threshold=0.6,
            max_models=3,
            max_strategies=2
        )
    
    @pytest.fixture
    def learner(self, learning_config):
        """返回学习器实例的fixture"""
        return Learner(learning_config)
    
    @pytest.fixture
    def sample_prediction_data(self):
        """返回样本预测数据的fixture"""
        return [
            {
                "model_id": "model1",
                "model_type": ModelType.DEEPSEEK,
                "prediction": "庄",
                "actual": "庄",
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat()
            },
            {
                "model_id": "model1",
                "model_type": ModelType.DEEPSEEK,
                "prediction": "闲",
                "actual": "闲",
                "confidence": 0.75,
                "timestamp": datetime.now().isoformat()
            },
            {
                "model_id": "model1",
                "model_type": ModelType.DEEPSEEK,
                "prediction": "庄",
                "actual": "闲",
                "confidence": 0.65,
                "timestamp": datetime.now().isoformat()
            },
            {
                "model_id": "model2",
                "model_type": ModelType.LLAMA,
                "prediction": "庄",
                "actual": "庄",
                "confidence": 0.90,
                "timestamp": datetime.now().isoformat()
            },
            {
                "model_id": "model2",
                "model_type": ModelType.LLAMA,
                "prediction": "闲",
                "actual": "庄",
                "confidence": 0.60,
                "timestamp": datetime.now().isoformat()
            }
        ]
    
    @pytest.fixture
    def sample_trade_data(self):
        """返回样本交易数据的fixture"""
        return [
            {
                "strategy_id": "strategy1",
                "strategy_type": StrategyType.FOLLOW,
                "profit": 100.0,
                "amount": 1000.0,
                "timestamp": datetime.now().isoformat()
            },
            {
                "strategy_id": "strategy1",
                "strategy_type": StrategyType.FOLLOW,
                "profit": -50.0,
                "amount": 1000.0,
                "timestamp": datetime.now().isoformat()
            },
            {
                "strategy_id": "strategy2",
                "strategy_type": StrategyType.REVERSE,
                "profit": 200.0,
                "amount": 2000.0,
                "timestamp": datetime.now().isoformat()
            },
            {
                "strategy_id": "strategy2",
                "strategy_type": StrategyType.REVERSE,
                "profit": -100.0,
                "amount": 1000.0,
                "timestamp": datetime.now().isoformat()
            }
        ]
    
    def test_init(self, learning_config):
        """测试初始化"""
        learner = Learner(learning_config)
        
        assert learner.config == learning_config
        assert learner.running is False
        assert isinstance(learner.last_learn_time, datetime)
        assert learner.model_performances == {}
        assert learner.strategy_performances == {}
        assert learner.prediction_buffer == []
        assert learner.trade_buffer == []
    
    @pytest.mark.asyncio
    async def test_start_enabled(self, learner):
        """测试启动功能(启用状态)"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        with patch.object(asyncio, 'create_task', new_callable=AsyncMock) as mock_create_task:
            with patch('asyncio.sleep', return_value=done_future):
                await learner.start()
                
                assert learner.running is True
                mock_create_task.assert_called_once()
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_start_disabled(self, learner):
        """测试启动功能(禁用状态)"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        learner.config.enabled = False
        with patch.object(asyncio, 'create_task', new_callable=AsyncMock) as mock_create_task:
            with patch('asyncio.sleep', return_value=done_future):
                await learner.start()
                
                assert learner.running is False
                mock_create_task.assert_not_called()
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_stop(self, learner):
        """测试停止功能"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 先启动
        learner.running = True
        
        # 然后停止
        with patch('asyncio.sleep', return_value=done_future):
            await learner.stop()
            
            assert learner.running is False
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_learning_loop(self, learner):
        """测试学习循环"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 设置为已经超过学习间隔
        learner.last_learn_time = datetime.now() - timedelta(seconds=4000)  
        learner.running = True
        
        # 模拟学习过程
        with patch.object(learner, '_perform_learning', new_callable=AsyncMock) as mock_learn:
            with patch('asyncio.sleep', return_value=done_future):
                # 设置超时保护
                try:
                    # 在运行一次后停止
                    def side_effect(*args, **kwargs):
                        learner.running = False
                        return done_future
                    
                    mock_learn.side_effect = side_effect
                    
                    await asyncio.wait_for(learner._learning_loop(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 强制停止
                    learner.running = False
                
                # 验证调用
                mock_learn.assert_called_once()
                assert (datetime.now() - learner.last_learn_time).total_seconds() < 10
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_learning_loop_exception(self, learner):
        """测试学习循环异常处理"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        learner.running = True
        
        # 设置测试条件：只运行一次循环
        def set_running_false(*args, **kwargs):
            learner.running = False
            return done_future
        
        # 模拟异常
        with patch.object(learner, '_perform_learning', side_effect=Exception("Test error")):
            with patch('asyncio.sleep', side_effect=set_running_false):
                await asyncio.wait_for(learner._learning_loop(), timeout=1.0)
                
                # 只要能执行到这里就表示异常被正确处理了
                assert True
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_perform_learning_insufficient_samples(self, learner):
        """测试执行学习 - 样本不足"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 不添加任何预测数据，确保样本数量不足
        
        # 模拟各个学习步骤
        with patch.object(learner, '_update_model_performance', new_callable=AsyncMock) as mock_update_model:
            with patch.object(learner, '_update_strategy_performance', new_callable=AsyncMock) as mock_update_strategy:
                with patch.object(learner, '_optimize_model_ensemble', new_callable=AsyncMock) as mock_optimize_model:
                    with patch.object(learner, '_optimize_strategy_ensemble', new_callable=AsyncMock) as mock_optimize_strategy:
                        with patch('asyncio.sleep', return_value=done_future):
                            await learner._perform_learning()
                            
                            # 验证结果 - 由于样本不足，不应调用任何方法
                            mock_update_model.assert_not_called()
                            mock_update_strategy.assert_not_called()
                            mock_optimize_model.assert_not_called()
                            mock_optimize_strategy.assert_not_called()
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_perform_learning_sufficient_samples(self, learner, sample_prediction_data):
        """测试执行学习 - 样本充足"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 添加足够的预测数据
        for data in sample_prediction_data:
            learner.prediction_buffer.append(data)
        
        # 确保样本充足
        learner.config.min_samples = 2
        
        # 模拟各个学习步骤
        with patch.object(learner, '_update_model_performance', new_callable=AsyncMock) as mock_update_model:
            with patch.object(learner, '_update_strategy_performance', new_callable=AsyncMock) as mock_update_strategy:
                with patch.object(learner, '_optimize_model_ensemble', new_callable=AsyncMock) as mock_optimize_model:
                    with patch.object(learner, '_optimize_strategy_ensemble', new_callable=AsyncMock) as mock_optimize_strategy:
                        with patch.object(learner, '_clear_buffers') as mock_clear:
                            with patch('asyncio.sleep', return_value=done_future):
                                await learner._perform_learning()
                                
                                # 验证调用
                                assert mock_update_model.called
                                assert mock_update_strategy.called
                                assert mock_optimize_model.called
                                assert mock_optimize_strategy.called
                                assert mock_clear.called
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_update_model_performance(self, learner, sample_prediction_data):
        """测试更新模型性能"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 添加预测数据
        for data in sample_prediction_data:
            learner.prediction_buffer.append(data)
        
        # 执行更新
        with patch('asyncio.sleep', return_value=done_future):
            await learner._update_model_performance()
        
        # 验证结果
        assert "model1" in learner.model_performances
        assert "model2" in learner.model_performances
        
        # 检查模型1的性能
        model1_perf = learner.model_performances["model1"]
        assert model1_perf.model_id == "model1"
        assert model1_perf.model_type == ModelType.DEEPSEEK
        assert model1_perf.predictions == 3
        assert model1_perf.accuracy > 0
        assert model1_perf.precision > 0
        assert model1_perf.recall >= 0
        assert model1_perf.f1 > 0
        
        # 检查模型2的性能
        model2_perf = learner.model_performances["model2"]
        assert model2_perf.model_id == "model2"
        assert model2_perf.model_type == ModelType.LLAMA
        assert model2_perf.predictions == 2
        assert model2_perf.accuracy > 0
        assert model2_perf.precision > 0
        assert model2_perf.recall >= 0
        assert model2_perf.f1 >= 0
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_update_strategy_performance(self, learner, sample_trade_data):
        """测试更新策略性能"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 添加交易数据
        for data in sample_trade_data:
            learner.trade_buffer.append(data)
        
        # 执行更新
        with patch('asyncio.sleep', return_value=done_future):
            await learner._update_strategy_performance()
        
        # 验证结果
        assert "strategy1" in learner.strategy_performances
        assert "strategy2" in learner.strategy_performances
        
        # 验证策略1的性能指标
        strategy1 = learner.strategy_performances["strategy1"]
        assert strategy1.strategy_id == "strategy1"
        assert strategy1.strategy_type == StrategyType.FOLLOW
        assert strategy1.win_rate == 0.5  # 1/2
        assert strategy1.profit_factor == 2.0  # 100/50
        assert strategy1.trades == 2
        
        # 验证策略2的性能指标
        strategy2 = learner.strategy_performances["strategy2"]
        assert strategy2.strategy_id == "strategy2"
        assert strategy2.strategy_type == StrategyType.REVERSE
        assert strategy2.win_rate == 0.5  # 1/2
        assert strategy2.profit_factor == 2.0  # 200/100
        assert strategy2.trades == 2
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_optimize_model_ensemble(self, learner):
        """测试优化模型组合"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 设置测试数据 - 3个好模型，1个差模型
        learner.model_performances = {
            "model1": ModelPerformance(
                model_id="model1",
                model_type=ModelType.DEEPSEEK,
                accuracy=0.8,
                precision=0.75,
                recall=0.7,
                f1=0.72,
                predictions=100
            ),
            "model2": ModelPerformance(
                model_id="model2",
                model_type=ModelType.LLAMA,
                accuracy=0.85,
                precision=0.8,
                recall=0.75,
                f1=0.77,
                predictions=150
            ),
            "model3": ModelPerformance(
                model_id="model3",
                model_type=ModelType.MISTRAL,
                accuracy=0.75,
                precision=0.7,
                recall=0.65,
                f1=0.67,
                predictions=120
            ),
            "model4": ModelPerformance(
                model_id="model4",
                model_type=ModelType.CNN_RNN,
                accuracy=0.55,  # 性能不达标
                precision=0.5,
                recall=0.45,
                f1=0.47,
                predictions=80
            )
        }
        
        # 执行优化
        with patch('asyncio.sleep', return_value=done_future):
            await learner._optimize_model_ensemble()
        
        # 验证结果 - 应该保留前3个模型(max_models=3)
        assert "model1" in learner.model_performances
        assert "model2" in learner.model_performances
        assert "model3" in learner.model_performances
        assert "model4" not in learner.model_performances  # 性能最差的应被移除
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    @pytest.mark.asyncio
    async def test_optimize_strategy_ensemble(self, learner):
        """测试优化策略组合"""
        # 创建一个完成的Future
        loop = asyncio.get_running_loop()
        done_future = loop.create_future()
        done_future.set_result(None)
        
        # 设置测试数据 - 2个好策略，1个差策略
        learner.strategy_performances = {
            "strategy1": StrategyPerformance(
                strategy_id="strategy1",
                strategy_type=StrategyType.FOLLOW,
                win_rate=0.6,
                profit_factor=1.8,
                trades=100
            ),
            "strategy2": StrategyPerformance(
                strategy_id="strategy2",
                strategy_type=StrategyType.REVERSE,
                win_rate=0.65,
                profit_factor=2.0,
                trades=150
            ),
            "strategy3": StrategyPerformance(
                strategy_id="strategy3",
                strategy_type=StrategyType.PATTERN,
                win_rate=0.4,  # 胜率不达标
                profit_factor=0.8,  # 盈亏比不达标
                trades=120
            )
        }
        
        # 执行优化
        with patch('asyncio.sleep', return_value=done_future):
            await learner._optimize_strategy_ensemble()
        
        # 验证结果 - 应该保留前2个策略(max_strategies=2)
        assert "strategy1" in learner.strategy_performances
        assert "strategy2" in learner.strategy_performances
        assert "strategy3" not in learner.strategy_performances  # 性能最差的应被移除
        
        # 确保测试清理资源
        await asyncio.sleep(0)
    
    def test_clear_buffers(self, learner, sample_prediction_data, sample_trade_data):
        """测试清理缓冲区"""
        # 添加数据
        for data in sample_prediction_data:
            learner.prediction_buffer.append(data)
        for data in sample_trade_data:
            learner.trade_buffer.append(data)
        
        # 执行清理
        learner._clear_buffers()
        
        # 验证结果
        assert len(learner.prediction_buffer) == 0
        assert len(learner.trade_buffer) == 0
    
    def test_add_prediction_result(self, learner):
        """测试添加预测结果"""
        # 添加预测结果
        prediction = {
            "model_id": "test_model",
            "model_type": ModelType.DEEPSEEK,
            "prediction": "庄",
            "actual": "庄",
            "confidence": 0.9
        }
        
        learner.add_prediction_result(prediction)
        
        # 验证结果
        assert len(learner.prediction_buffer) == 1
        assert learner.prediction_buffer[0]["model_id"] == "test_model"
        assert "timestamp" in learner.prediction_buffer[0]  # 应该自动添加时间戳
    
    def test_add_trade_result(self, learner):
        """测试添加交易结果"""
        # 添加交易结果
        trade = {
            "strategy_id": "test_strategy",
            "strategy_type": StrategyType.FOLLOW,
            "profit": 100.0,
            "amount": 1000.0
        }
        
        learner.add_trade_result(trade)
        
        # 验证结果
        assert len(learner.trade_buffer) == 1
        assert learner.trade_buffer[0]["strategy_id"] == "test_strategy"
        assert "timestamp" in learner.trade_buffer[0]  # 应该自动添加时间戳
    
    def test_get_best_model(self, learner):
        """测试获取最佳模型"""
        # 设置测试数据
        learner.model_performances = {
            "model1": ModelPerformance(
                model_id="model1",
                model_type=ModelType.DEEPSEEK,
                accuracy=0.8,
                f1=0.75
            ),
            "model2": ModelPerformance(
                model_id="model2",
                model_type=ModelType.LLAMA,
                accuracy=0.85,  # 最高准确率
                f1=0.8  # 最高F1
            )
        }
        
        # 获取最佳模型
        best_model = learner.get_best_model()
        
        # 验证结果
        assert best_model is not None
        assert best_model[0] == "model2"  # model2性能更好
    
    def test_get_best_model_empty(self, learner):
        """测试无模型时获取最佳模型"""
        # 不设置任何模型性能数据
        
        # 获取最佳模型
        best_model = learner.get_best_model()
        
        # 验证结果
        assert best_model is None
    
    def test_get_best_strategy(self, learner):
        """测试获取最佳策略"""
        # 设置测试数据
        learner.strategy_performances = {
            "strategy1": StrategyPerformance(
                strategy_id="strategy1",
                strategy_type=StrategyType.FOLLOW,
                win_rate=0.6,
                profit_factor=1.8
            ),
            "strategy2": StrategyPerformance(
                strategy_id="strategy2",
                strategy_type=StrategyType.REVERSE,
                win_rate=0.65,  # 最高胜率
                profit_factor=2.0  # 最高盈亏比
            )
        }
        
        # 获取最佳策略
        best_strategy = learner.get_best_strategy()
        
        # 验证结果
        assert best_strategy is not None
        assert best_strategy[0] == "strategy2"  # strategy2性能更好
    
    def test_get_best_strategy_empty(self, learner):
        """测试无策略时获取最佳策略"""
        # 不设置任何策略性能数据
        
        # 获取最佳策略
        best_strategy = learner.get_best_strategy()
        
        # 验证结果
        assert best_strategy is None
    
    def test_get_learning_stats(self, learner):
        """测试获取学习统计"""
        # 设置测试数据
        learner.model_performances = {
            "model1": ModelPerformance(
                model_id="model1",
                model_type=ModelType.DEEPSEEK,
                accuracy=0.8
            )
        }
        learner.strategy_performances = {
            "strategy1": StrategyPerformance(
                strategy_id="strategy1",
                strategy_type=StrategyType.FOLLOW,
                win_rate=0.6
            )
        }
        learner.prediction_buffer = [{} for _ in range(5)]
        learner.trade_buffer = [{} for _ in range(3)]
        
        # 获取统计信息
        stats = learner.get_learning_stats()
        
        # 验证结果
        assert stats["models_count"] == 1
        assert stats["strategies_count"] == 1
        assert stats["prediction_samples"] == 5
        assert stats["trade_samples"] == 3
        assert "best_model" in stats
        assert "best_strategy" in stats 