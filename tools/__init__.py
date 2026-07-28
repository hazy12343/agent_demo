"""
工具模块
"""
from .base import BaseTool, ToolRegistry, ToolParameter, tool_registry
from .calculator import CalculatorTool
from .search import SearchTool
from .weather import WeatherTool


def register_all_tools():
    """注册所有工具"""
    tool_registry.register(CalculatorTool())
    tool_registry.register(SearchTool())
    tool_registry.register(WeatherTool())


__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolParameter",
    "tool_registry",
    "register_all_tools",
    "CalculatorTool",
    "SearchTool",
    "WeatherTool",
]
