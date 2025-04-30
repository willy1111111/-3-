#!/usr/bin/env python3
"""
百家乐AI系统 - 参数一致性检查工具
检查所有服务中参数的定义是否一致
"""

import asyncio
import logging
import json
import os
import sys
import aiohttp
from datetime import datetime
from typing import Dict, List, Any, Set
from tabulate import tabulate

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("parameter_check.log")
    ]
)
logger = logging.getLogger("ParameterChecker")

# 服务定义
SERVICES = {
    "baccarat_ai": {
        "host": "localhost", 
        "port": 8001, 
        "endpoints": ["/predict", "/bet", "/bet/result"]
    },
    "trading": {
        "host": "localhost", 
        "port": 8003, 
        "endpoints": ["/trade", "/trades", "/trade/batch"]
    },
    "risk": {
        "host": "localhost", 
        "port": 8007, 
        "endpoints": ["/risk/assess", "/risk/profile/{user_id}"]
    }
}

class ParameterChecker:
    """参数一致性检查器"""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "service_parameters": {},
            "inconsistencies": [],
            "suggestions": []
        }
    
    async def check_all(self):
        """检查所有服务的参数一致性"""
        logger.info("开始参数一致性检查")
        
        # 1. 获取服务API文档
        await self.retrieve_api_docs()
        
        # 2. 提取参数定义
        self.extract_parameters()
        
        # 3. 分析一致性
        self.analyze_consistency()
        
        # 4. 生成报告
        self.generate_report()
        
        logger.info("参数一致性检查完成")
    
    async def retrieve_api_docs(self):
        """获取API文档"""
        logger.info("获取API文档")
        
        service_parameters = {}
        
        # 获取百家乐AI服务文档
        try:
            logger.info("获取百家乐AI服务API文档")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8001/openapi.json",
                    timeout=5
                ) as response:
                    if response.status == 200:
                        # OpenAPI文档获取成功
                        api_docs = await response.json()
                        service_parameters["baccarat_ai"] = {
                            "source": "openapi",
                            "raw_data": api_docs
                        }
                        logger.info("获取百家乐AI服务OpenAPI文档成功")
                    else:
                        # 获取失败，手动定义
                        logger.warning(f"获取百家乐AI服务OpenAPI文档失败: {response.status}")
                        service_parameters["baccarat_ai"] = {
                            "source": "manual",
                            "endpoints": {
                                "/predict": {
                                    "post": {
                                        "parameters": []
                                    }
                                }
                            }
                        }
        except Exception as e:
            logger.error(f"获取百家乐AI服务API文档出错: {str(e)}")
            service_parameters["baccarat_ai"] = {
                "source": "error",
                "error": str(e)
            }
        
        # 获取交易服务文档
        try:
            logger.info("获取交易服务API文档")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8003/openapi.json",
                    timeout=5
                ) as response:
                    if response.status == 200:
                        # OpenAPI文档获取成功
                        api_docs = await response.json()
                        service_parameters["trading"] = {
                            "source": "openapi",
                            "raw_data": api_docs
                        }
                        logger.info("获取交易服务OpenAPI文档成功")
                    else:
                        # 获取失败，手动定义
                        logger.warning(f"获取交易服务OpenAPI文档失败: {response.status}")
                        service_parameters["trading"] = {
                            "source": "manual",
                            "endpoints": {}
                        }
                        # 对每个端点进行分析
                        for endpoint in SERVICES["trading"]["endpoints"]:
                            endpoint_url = f"http://localhost:8003{endpoint}"
                            service_parameters["trading"]["endpoints"][endpoint] = {
                                "url": endpoint_url,
                                "methods": ["GET", "POST"],  # 假设所有端点都支持这两种方法
                                "parameters": {}
                            }
                        logger.info("手动分析交易服务API端点")
        except Exception as e:
            logger.error(f"获取交易服务API文档失败: {str(e)}")
            service_parameters["trading"] = {
                "source": "error",
                "error": str(e)
            }
        
        # 获取风控服务文档
        try:
            logger.info("获取风控服务API文档")
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://localhost:8007/openapi.json",
                    timeout=5
                ) as response:
                    if response.status == 200:
                        # OpenAPI文档获取成功
                        api_docs = await response.json()
                        service_parameters["risk"] = {
                            "source": "openapi",
                            "raw_data": api_docs
                        }
                        logger.info("获取风控服务OpenAPI文档成功")
                    else:
                        # 获取失败，手动定义
                        logger.warning(f"获取风控服务OpenAPI文档失败: {response.status}")
                        service_parameters["risk"] = {
                            "source": "manual",
                            "endpoints": {}
                        }
                        # 对每个端点进行分析
                        for endpoint in SERVICES["risk"]["endpoints"]:
                            endpoint_url = f"http://localhost:8007{endpoint}"
                            service_parameters["risk"]["endpoints"][endpoint] = {
                                "url": endpoint_url,
                                "methods": ["GET", "POST"],  # 假设所有端点都支持这两种方法
                                "parameters": {}
                            }
                        logger.info("手动分析风控服务API端点")
        except Exception as e:
            logger.error(f"获取风控服务API文档失败: {str(e)}")
            service_parameters["risk"] = {
                "source": "error",
                "error": str(e)
            }
        
        self.results["service_parameters"] = service_parameters
    
    def extract_parameters(self):
        """提取参数定义"""
        logger.info("提取参数定义")
        
        # 为演示目的，这里添加一些示例参数定义
        # 在实际使用中，应该从API文档中提取
        
        # 百家乐AI服务参数
        if "baccarat_ai" in self.results["service_parameters"]:
            self.results["service_parameters"]["baccarat_ai"]["parameters"] = {
                "history": {
                    "type": "array",
                    "item_type": "string",
                    "required": True,
                    "description": "历史记录列表",
                    "example": ["庄", "闲", "庄"]
                },
                "model_version": {
                    "type": "string",
                    "required": False,
                    "default": "DeepSeek-7B-v2",
                    "description": "模型版本"
                },
                "confidence": {
                    "type": "float",
                    "required": True,
                    "description": "预测置信度"
                },
                "table_id": {
                    "type": "string",
                    "required": True,
                    "description": "台面ID"
                },
                "round_id": {
                    "type": "string",
                    "required": True,
                    "description": "局号ID"
                },
                "success": {
                    "type": "boolean",
                    "required": True,
                    "description": "是否成功"
                },
                "payout": {
                    "type": "float",
                    "required": True,
                    "description": "收益"
                }
            }
        
        # 交易服务参数
        if "trading" in self.results["service_parameters"]:
            self.results["service_parameters"]["trading"]["parameters"] = {
                "user_id": {
                    "type": "string",
                    "required": True,
                    "description": "用户ID"
                },
                "game_id": {
                    "type": "string",
                    "required": True,
                    "description": "游戏ID"
                },
                "bet_type": {
                    "type": "string",
                    "required": True,
                    "description": "投注类型",
                    "enum": ["闲", "庄", "和"]
                },
                "amount": {
                    "type": "float",
                    "required": True,
                    "description": "投注金额"
                },
                "odds": {
                    "type": "float",
                    "required": True,
                    "description": "赔率"
                },
                "strategy_id": {
                    "type": "string",
                    "required": False,
                    "description": "策略ID"
                },
                "auto_mode": {
                    "type": "boolean",
                    "required": False,
                    "default": False,
                    "description": "是否自动模式"
                },
                "hall_index": {
                    "type": "integer",
                    "required": True,
                    "description": "大厅索引"
                },
                "number": {
                    "type": "string",
                    "required": True,
                    "description": "局号"
                },
                "round_id": {
                    "type": "string",
                    "required": False,
                    "description": "局号ID"
                }
            }
        
        # 风控服务参数
        if "risk" in self.results["service_parameters"]:
            self.results["service_parameters"]["risk"]["parameters"] = {
                "user_id": {
                    "type": "string",
                    "required": True,
                    "description": "用户ID"
                },
                "bet_amount": {
                    "type": "float",
                    "required": True,
                    "description": "投注金额"
                },
                "game_id": {
                    "type": "string",
                    "required": True,
                    "description": "游戏ID"
                },
                "bet_type": {
                    "type": "string",
                    "required": True,
                    "description": "投注类型"
                },
                "odds": {
                    "type": "float",
                    "required": True,
                    "description": "赔率"
                },
                "amount": {
                    "type": "float",
                    "required": False,
                    "description": "投注金额（冗余字段）"
                }
            }
    
    def analyze_consistency(self):
        """分析参数的一致性"""
        logger.info("分析参数一致性")
        
        # 1. 收集所有参数名称
        all_params = set()
        for service_name, service_data in self.results["service_parameters"].items():
            if "parameters" in service_data:
                all_params.update(service_data["parameters"].keys())
        
        # 2. 检查每个参数在不同服务中的定义是否一致
        for param in all_params:
            param_definitions = {}
            
            for service_name, service_data in self.results["service_parameters"].items():
                if "parameters" in service_data and param in service_data["parameters"]:
                    param_definitions[service_name] = service_data["parameters"][param]
            
            # 如果参数在多个服务中都有定义
            if len(param_definitions) > 1:
                # 检查类型一致性
                types = set(definition["type"] for definition in param_definitions.values())
                if len(types) > 1:
                    self.results["inconsistencies"].append({
                        "parameter": param,
                        "issue": "类型不一致",
                        "details": {service: definition["type"] for service, definition in param_definitions.items()}
                    })
                    
                    # 添加建议
                    self.results["suggestions"].append({
                        "parameter": param,
                        "suggestion": f"统一参数 {param} 的类型为 {next(iter(types))}"
                    })
                
                # 检查必填性一致性
                required = set(definition.get("required", False) for definition in param_definitions.values())
                if len(required) > 1:
                    self.results["inconsistencies"].append({
                        "parameter": param,
                        "issue": "必填性不一致",
                        "details": {service: definition.get("required", False) for service, definition in param_definitions.items()}
                    })
                    
                    # 添加建议
                    most_strict = True in required
                    self.results["suggestions"].append({
                        "parameter": param,
                        "suggestion": f"统一参数 {param} 的必填性为 {most_strict}"
                    })
        
        # 3. 检查参数命名一致性
        naming_inconsistencies = []
        
        # 检查常见的命名不一致情况
        if "amount" in all_params and "bet_amount" in all_params:
            naming_inconsistencies.append({
                "parameters": ["amount", "bet_amount"],
                "suggestion": "统一使用 amount 表示投注金额"
            })
        
        if "round_id" in all_params and "number" in all_params:
            naming_inconsistencies.append({
                "parameters": ["round_id", "number"],
                "suggestion": "统一使用 round_id 表示局号"
            })
        
        # 添加到结果中
        self.results["naming_inconsistencies"] = naming_inconsistencies
    
    def generate_report(self):
        """生成报告"""
        logger.info("生成参数一致性检查报告")
        
        # 保存报告到文件
        with open("parameter_check_report.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # 打印一致性问题
        if self.results["inconsistencies"]:
            logger.info(f"发现 {len(self.results['inconsistencies'])} 个参数一致性问题:")
            
            table_data = []
            for issue in self.results["inconsistencies"]:
                details = ", ".join(f"{service}: {value}" for service, value in issue["details"].items())
                table_data.append([issue["parameter"], issue["issue"], details])
            
            table = tabulate(
                table_data,
                headers=["参数", "问题", "详情"],
                tablefmt="grid"
            )
            print(table)
        else:
            logger.info("未发现参数一致性问题")
        
        # 打印命名不一致问题
        if "naming_inconsistencies" in self.results and self.results["naming_inconsistencies"]:
            logger.info(f"发现 {len(self.results['naming_inconsistencies'])} 个参数命名不一致问题:")
            
            table_data = []
            for issue in self.results["naming_inconsistencies"]:
                table_data.append([", ".join(issue["parameters"]), issue["suggestion"]])
            
            table = tabulate(
                table_data,
                headers=["参数", "建议"],
                tablefmt="grid"
            )
            print(table)
        else:
            logger.info("未发现参数命名不一致问题")
        
        # 打印建议
        if self.results["suggestions"]:
            logger.info(f"提供 {len(self.results['suggestions'])} 个改进建议:")
            
            table_data = []
            for suggestion in self.results["suggestions"]:
                table_data.append([suggestion["parameter"], suggestion["suggestion"]])
            
            table = tabulate(
                table_data,
                headers=["参数", "建议"],
                tablefmt="grid"
            )
            print(table)
        
        logger.info(f"详细报告已保存到 parameter_check_report.json")

async def main():
    """主函数"""
    try:
        checker = ParameterChecker()
        await checker.check_all()
    except Exception as e:
        logger.error(f"参数检查过程中出现未处理异常: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 