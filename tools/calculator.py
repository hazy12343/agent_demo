"""
计算器工具
"""
from typing import Any, Dict
from .base import BaseTool, ToolParameter


class CalculatorTool(BaseTool):
    """计算器工具 - 支持基本数学运算"""
    
    def __init__(self):
        super().__init__()
        self.name = "calculator"
        self.description = "执行基本数学运算，支持加减乘除和幂运算"
        self.parameters = [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式，例如: '2 + 3 * 4' 或 '10 / 2'"
            )
        ]
    
    def execute(self, expression: str) -> Dict[str, Any]:
        """执行计算"""
        try:
            # 安全地评估表达式（只允许数字和基本运算符）
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return {
                    "success": False,
                    "error": "表达式包含不允许的字符"
                }
            
            result = eval(expression, {"__builtins__": {}}, {})
            
            return {
                "success": True,
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"计算错误: {str(e)}"
            }
