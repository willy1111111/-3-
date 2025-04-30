import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import json
from datetime import datetime

from microservices.baccarat_ai.parameter_checker import (
    ParameterChecker,
    SERVICES
)

class TestParameterChecker:
    """测试百家乐AI参数一致性检查器功能"""
    
    @pytest.fixture
    def parameter_checker(self):
        """返回参数检查器实例的fixture"""
        return ParameterChecker()
    
    @pytest.fixture
    def mock_openapi_response(self):
        """模拟OpenAPI响应的fixture"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "openapi": "3.0.0",
            "info": {"title": "测试API", "version": "1.0.0"},
            "paths": {
                "/predict": {
                    "post": {
                        "parameters": [],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "history": {
                                                "type": "array",
                                                "items": {"type": "string"}
                                            },
                                            "confidence": {"type": "number"},
                                            "model_version": {"type": "string"}
                                        },
                                        "required": ["history", "confidence"]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return mock_response
    
    @pytest.fixture
    def mock_failed_response(self):
        """模拟失败的API响应的fixture"""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.json.side_effect = Exception("Invalid JSON")
        return mock_response
    
    @pytest.mark.asyncio
    async def test_retrieve_api_docs_success(self, parameter_checker, mock_openapi_response):
        """测试成功获取API文档"""
        # 模拟HTTP请求
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_openapi_response
            
            # 执行文档获取
            await parameter_checker.retrieve_api_docs()
            
            # 验证结果
            assert "baccarat_ai" in parameter_checker.results["service_parameters"]
            assert parameter_checker.results["service_parameters"]["baccarat_ai"]["source"] == "openapi"
            assert "raw_data" in parameter_checker.results["service_parameters"]["baccarat_ai"]
    
    @pytest.mark.asyncio
    async def test_retrieve_api_docs_failure(self, parameter_checker, mock_failed_response):
        """测试获取API文档失败后的手动分析"""
        # 模拟HTTP请求失败
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_failed_response
            
            # 执行文档获取
            await parameter_checker.retrieve_api_docs()
            
            # 验证结果
            assert "baccarat_ai" in parameter_checker.results["service_parameters"]
            assert parameter_checker.results["service_parameters"]["baccarat_ai"]["source"] == "manual"
            assert "endpoints" in parameter_checker.results["service_parameters"]["baccarat_ai"]
    
    @pytest.mark.asyncio
    async def test_retrieve_api_docs_exception(self, parameter_checker):
        """测试获取API文档时发生异常"""
        # 模拟HTTP请求异常
        with patch('aiohttp.ClientSession.get', side_effect=Exception("Connection error")):
            # 执行文档获取
            await parameter_checker.retrieve_api_docs()
            
            # 验证结果
            assert "baccarat_ai" in parameter_checker.results["service_parameters"]
            assert parameter_checker.results["service_parameters"]["baccarat_ai"]["source"] == "error"
            assert "error" in parameter_checker.results["service_parameters"]["baccarat_ai"]
    
    def test_extract_parameters(self, parameter_checker):
        """测试提取参数定义"""
        # 设置初始状态
        parameter_checker.results["service_parameters"] = {
            "baccarat_ai": {"source": "manual", "parameters": {}},
            "trading": {"source": "manual", "parameters": {}},
            "risk": {"source": "manual", "parameters": {}}
        }
        
        # 执行参数提取
        parameter_checker.extract_parameters()
        
        # 验证结果
        assert "history" in parameter_checker.results["service_parameters"]["baccarat_ai"]["parameters"]
        assert "user_id" in parameter_checker.results["service_parameters"]["trading"]["parameters"]
        assert "user_id" in parameter_checker.results["service_parameters"]["risk"]["parameters"]
    
    def test_analyze_consistency(self, parameter_checker):
        """测试一致性分析"""
        # 设置初始状态 - 创建具有不一致性的参数
        parameter_checker.results["service_parameters"] = {
            "baccarat_ai": {
                "parameters": {
                    "round_id": {
                        "type": "string", 
                        "required": True,
                        "description": "局号ID"
                    }
                }
            },
            "trading": {
                "parameters": {
                    "round_id": {
                        "type": "string", 
                        "required": False,  # 不一致的required属性
                        "description": "局号ID"
                    }
                }
            }
        }
        
        # 执行一致性分析
        parameter_checker.analyze_consistency()
        
        # 验证结果
        assert len(parameter_checker.results["inconsistencies"]) > 0
        # 至少有一个关于round_id的不一致
        found = False
        for item in parameter_checker.results["inconsistencies"]:
            if item["parameter"] == "round_id" and item["issue"] == "必填性不一致":
                found = True
                break
        assert found, "应该检测到round_id的必填性不一致"
    
    def test_generate_report(self, parameter_checker):
        """测试生成报告"""
        # 设置初始状态
        parameter_checker.results = {
            "timestamp": datetime.now().isoformat(),
            "service_parameters": {
                "baccarat_ai": {"parameters": {"p1": {}}},
                "trading": {"parameters": {"p2": {}}}
            },
            "inconsistencies": [
                {
                    "parameter": "test_param", 
                    "issue": "类型不一致", 
                    "details": {"s1": "string", "s2": "integer"}
                }
            ],
            "suggestions": [
                {
                    "parameter": "test_param",
                    "suggestion": "统一参数类型为string"
                }
            ]
        }
        
        # 模拟文件写入
        with patch('builtins.open', MagicMock()):
            with patch('json.dump') as mock_dump:
                # 执行报告生成
                parameter_checker.generate_report()
                
                # 验证调用
                assert mock_dump.called
    
    @pytest.mark.asyncio
    async def test_check_all(self, parameter_checker):
        """测试完整检查流程"""
        # 模拟各个阶段
        with patch.object(parameter_checker, 'retrieve_api_docs', return_value=None) as mock_retrieve:
            with patch.object(parameter_checker, 'extract_parameters') as mock_extract:
                with patch.object(parameter_checker, 'analyze_consistency') as mock_analyze:
                    with patch.object(parameter_checker, 'generate_report') as mock_report:
                        # 执行完整检查
                        await parameter_checker.check_all()
                        
                        # 验证所有步骤都被调用
                        mock_retrieve.assert_called_once()
                        mock_extract.assert_called_once()
                        mock_analyze.assert_called_once()
                        mock_report.assert_called_once()

@pytest.mark.asyncio
async def test_main():
    """测试main函数"""
    with patch('microservices.baccarat_ai.parameter_checker.ParameterChecker') as MockChecker:
        # 设置mock
        instance = MockChecker.return_value
        instance.check_all = AsyncMock()
        
        # 调用main函数
        with patch('asyncio.run') as mock_run:
            from microservices.baccarat_ai.parameter_checker import main
            await main()
            
            # 验证调用
            instance.check_all.assert_called_once()

