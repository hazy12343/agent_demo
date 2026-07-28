"""
测试工具模块
测试 calculator、search、weather 工具以及工具注册机制
"""
import pytest
from tools.base import BaseTool, ToolParameter, ToolRegistry
from tools.calculator import CalculatorTool
from tools.search import SearchTool
from tools.weather import WeatherTool


# ============ 计算器工具测试 ============

class TestCalculatorTool:
    """计算器工具测试"""
    
    def setup_method(self):
        self.tool = CalculatorTool()
    
    def test_basic_addition(self):
        result = self.tool.execute(expression="2 + 3")
        assert result["success"] is True
        assert result["result"] == 5
    
    def test_basic_subtraction(self):
        result = self.tool.execute(expression="10 - 4")
        assert result["success"] is True
        assert result["result"] == 6
    
    def test_basic_multiplication(self):
        result = self.tool.execute(expression="3 * 7")
        assert result["success"] is True
        assert result["result"] == 21
    
    def test_basic_division(self):
        result = self.tool.execute(expression="15 / 3")
        assert result["success"] is True
        assert result["result"] == 5.0
    
    def test_complex_expression(self):
        result = self.tool.execute(expression="(2 + 3) * 4 - 1")
        assert result["success"] is True
        assert result["result"] == 19
    
    def test_float_calculation(self):
        result = self.tool.execute(expression="3.14 * 2")
        assert result["success"] is True
        assert abs(result["result"] - 6.28) < 0.01
    
    def test_invalid_expression(self):
        result = self.tool.execute(expression="2 3 +")
        assert result["success"] is False
        assert "error" in result
    
    def test_disallowed_characters(self):
        result = self.tool.execute(expression="import os")
        assert result["success"] is False
    
    def test_schema_generation(self):
        schema = self.tool.get_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert "expression" in schema["function"]["parameters"]["properties"]


# ============ 搜索工具测试 ============

class TestSearchTool:
    """搜索工具测试"""
    
    def setup_method(self):
        self.tool = SearchTool()
    
    def test_search_python(self):
        result = self.tool.execute(query="python 教程")
        assert result["success"] is True
        assert result["count"] > 0
        assert any("Python" in r["title"] for r in result["results"])
    
    def test_search_agent(self):
        result = self.tool.execute(query="AI agent 架构")
        assert result["success"] is True
        assert result["count"] > 0
    
    def test_search_llm(self):
        result = self.tool.execute(query="大语言模型 LLM")
        assert result["success"] is True
        assert result["count"] > 0
    
    def test_search_no_match(self):
        result = self.tool.execute(query="不存在的关键词")
        assert result["success"] is True
        assert result["count"] > 0  # 仍返回通用结果
    
    def test_schema_generation(self):
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]


# ============ 天气工具测试 ============

class TestWeatherTool:
    """天气工具测试"""
    
    def setup_method(self):
        self.tool = WeatherTool()
    
    def test_query_beijing(self):
        result = self.tool.execute(city="北京")
        assert result["success"] is True
        assert result["city"] == "北京"
        assert "temperature" in result
        assert "condition" in result
        assert "suggestion" in result
    
    def test_query_shanghai(self):
        result = self.tool.execute(city="上海")
        assert result["success"] is True
        assert result["city"] == "上海"
    
    def test_query_guangzhou_rain(self):
        result = self.tool.execute(city="广州")
        assert result["success"] is True
        assert "雨" in result["condition"] or "Rain" in result["condition"]
        assert "带伞" in result["suggestion"]
    
    def test_query_new_york(self):
        result = self.tool.execute(city="New York")
        assert result["success"] is True
    
    def test_query_unknown_city(self):
        result = self.tool.execute(city="不存在的城市")
        assert result["success"] is False
        assert "error" in result
        assert "available_cities" in result
    
    def test_schema_generation(self):
        schema = self.tool.get_schema()
        assert schema["function"]["name"] == "weather"
        assert "city" in schema["function"]["parameters"]["properties"]


# ============ 工具注册机制测试 ============

class TestToolRegistry:
    """工具注册表测试"""
    
    def setup_method(self):
        self.registry = ToolRegistry()
    
    def test_register_tool(self):
        tool = CalculatorTool()
        self.registry.register(tool)
        assert "calculator" in self.registry.list_tools()
    
    def test_get_tool(self):
        tool = CalculatorTool()
        self.registry.register(tool)
        retrieved = self.registry.get("calculator")
        assert retrieved is not None
        assert retrieved.name == "calculator"
    
    def test_get_nonexistent_tool(self):
        result = self.registry.get("nonexistent")
        assert result is None
    
    def test_execute_tool(self):
        tool = CalculatorTool()
        self.registry.register(tool)
        result = self.registry.execute("calculator", expression="1 + 1")
        assert result["success"] is True
        assert result["result"] == 2
    
    def test_execute_nonexistent_tool(self):
        with pytest.raises(ValueError, match="工具不存在"):
            self.registry.execute("nonexistent")
    
    def test_get_all_schemas(self):
        self.registry.register(CalculatorTool())
        self.registry.register(SearchTool())
        self.registry.register(WeatherTool())
        
        schemas = self.registry.get_all_schemas()
        assert len(schemas) == 3
        
        names = [s["function"]["name"] for s in schemas]
        assert "calculator" in names
        assert "search" in names
        assert "weather" in names
    
    def test_validate_missing_required_param(self):
        tool = CalculatorTool()
        self.registry.register(tool)
        with pytest.raises(ValueError, match="缺少必需参数"):
            self.registry.execute("calculator")  # 不传 expression
    
    def test_list_tools(self):
        self.registry.register(CalculatorTool())
        self.registry.register(SearchTool())
        
        tools = self.registry.list_tools()
        assert len(tools) == 2
        assert "calculator" in tools
        assert "search" in tools
