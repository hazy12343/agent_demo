"""
测试 LLM 输出解析模块
"""
import pytest
from parser import OutputParser, ParsedOutput, ToolCall


class TestOutputParser:
    """输出解析器测试"""
    
    def setup_method(self):
        self.parser = OutputParser()
    
    # ============ Function Calling 模式测试 ============
    
    def test_parse_tool_call(self):
        """测试解析标准 function calling 响应"""
        response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "name": "calculator",
                    "arguments": {"expression": "2 + 3"}
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}
        }
        
        result = self.parser.parse(response)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"expression": "2 + 3"}
        assert result.tool_calls[0].tool_call_id == "call_001"
    
    def test_parse_multiple_tool_calls(self):
        """测试解析多个工具调用"""
        response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_001",
                    "name": "weather",
                    "arguments": {"city": "北京"}
                },
                {
                    "id": "call_002",
                    "name": "calculator",
                    "arguments": {"expression": "22 * 2"}
                }
            ],
            "finish_reason": "tool_calls"
        }
        
        result = self.parser.parse(response)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "weather"
        assert result.tool_calls[1].name == "calculator"
    
    def test_parse_tool_call_with_thinking(self):
        """测试工具调用时附带的思考过程"""
        response = {
            "content": "用户想知道北京的天气，我来调用天气工具。",
            "tool_calls": [
                {
                    "id": "call_001",
                    "name": "weather",
                    "arguments": {"city": "北京"}
                }
            ],
            "finish_reason": "tool_calls"
        }
        
        result = self.parser.parse(response)
        assert len(result.tool_calls) == 1
        assert result.thinking == "用户想知道北京的天气，我来调用天气工具。"
    
    # ============ 纯文本模式测试 ============
    
    def test_parse_direct_response(self):
        """测试直接文本回复"""
        response = {
            "content": "你好！有什么可以帮助你的吗？",
            "tool_calls": None,
            "finish_reason": "stop"
        }
        
        result = self.parser.parse(response)
        assert result.final_answer == "你好！有什么可以帮助你的吗？"
        assert len(result.tool_calls) == 0
    
    def test_parse_response_with_thinking_tags(self):
        """测试带 <thinking> 标签的回复"""
        response = {
            "content": "<thinking>用户问的是简单的问候问题</thinking>\n你好！很高兴为你服务。",
            "tool_calls": None,
            "finish_reason": "stop"
        }
        
        result = self.parser.parse(response)
        assert result.thinking == "用户问的是简单的问候问题"
        assert result.final_answer == "你好！很高兴为你服务。"
    
    def test_parse_empty_content(self):
        """测试空内容"""
        response = {
            "content": "",
            "tool_calls": None,
            "finish_reason": "stop"
        }
        
        result = self.parser.parse(response)
        assert result.final_answer is None
        assert len(result.tool_calls) == 0
    
    # ============ 格式化工具结果测试 ============
    
    def test_format_tool_result(self):
        """测试格式化工具执行结果"""
        result = self.parser.format_tool_result(
            tool_call_id="call_001",
            tool_name="calculator",
            result={"success": True, "result": 5}
        )
        
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_001"
        assert '"success": true' in result["content"]
        assert '"result": 5' in result["content"]


class TestToolCall:
    """ToolCall 数据类测试"""
    
    def test_create_tool_call(self):
        tc = ToolCall(
            tool_call_id="call_123",
            name="search",
            arguments={"query": "python"}
        )
        assert tc.tool_call_id == "call_123"
        assert tc.name == "search"
        assert tc.arguments["query"] == "python"


class TestParsedOutput:
    """ParsedOutput 数据类测试"""
    
    def test_default_values(self):
        po = ParsedOutput()
        assert po.thinking is None
        assert po.tool_calls == []
        assert po.final_answer is None
        assert po.raw_response == {}
    
    def test_with_values(self):
        po = ParsedOutput(
            thinking="思考中...",
            final_answer="最终答案",
            raw_response={"content": "test"}
        )
        assert po.thinking == "思考中..."
        assert po.final_answer == "最终答案"
