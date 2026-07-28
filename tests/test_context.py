"""
测试 Context 管理模块
"""
import pytest
from context import ContextManager, Message


class TestContextManager:
    """Context 管理器测试"""
    
    def setup_method(self):
        self.ctx = ContextManager(
            max_turns=5,
            compression_threshold=1000,
            max_context_tokens=2000
        )
        self.ctx.set_system_prompt("你是一个智能助手。")
    
    # ============ 基本消息操作测试 ============
    
    def test_add_user_message(self):
        msg = self.ctx.add_user_message("你好")
        assert msg.role == "user"
        assert msg.content == "你好"
        assert len(self.ctx.messages) == 1
    
    def test_add_assistant_message(self):
        msg = self.ctx.add_assistant_message("你好！有什么可以帮助你的？")
        assert msg.role == "assistant"
        assert msg.content == "你好！有什么可以帮助你的？"
    
    def test_add_tool_message(self):
        msg = self.ctx.add_tool_message(
            tool_call_id="call_001",
            tool_name="calculator",
            result={"success": True, "result": 5}
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_001"
        assert msg.metadata["tool_name"] == "calculator"
    
    def test_add_assistant_with_tool_calls(self):
        tool_calls = [
            {
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "weather",
                    "arguments": '{"city": "北京"}'
                }
            }
        ]
        msg = self.ctx.add_assistant_message(
            content="让我查一下天气",
            tool_calls=tool_calls,
            thinking="用户想知道天气"
        )
        assert msg.tool_calls == tool_calls
        assert msg.metadata["thinking"] == "用户想知道天气"
    
    # ============ 消息格式测试 ============
    
    def test_get_messages_format(self):
        """测试消息的 OpenAI API 格式"""
        self.ctx.add_user_message("你好")
        self.ctx.add_assistant_message("你好！")
        
        messages = self.ctx.get_messages()
        
        assert len(messages) == 3  # system + user + assistant
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "你是一个智能助手。"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "你好"
        assert messages[2]["role"] == "assistant"
    
    def test_get_messages_with_tool_calls(self):
        """测试包含工具调用的消息格式"""
        self.ctx.add_user_message("北京天气怎么样？")
        
        tool_calls = [{
            "id": "call_001",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city": "北京"}'}
        }]
        self.ctx.add_assistant_message(content="", tool_calls=tool_calls)
        self.ctx.add_tool_message("call_001", "weather", {"temp": "22°C"})
        
        messages = self.ctx.get_messages()
        
        # 验证 tool_calls 格式
        assistant_msg = messages[2]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["id"] == "call_001"
        
        # 验证 tool result 格式
        tool_msg = messages[3]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_001"
    
    # ============ 轮次限制测试 ============
    
    def test_turn_count(self):
        """测试轮次计数"""
        self.ctx.add_user_message("问题1")
        assert self.ctx.get_turn_count() == 1
        
        self.ctx.add_assistant_message("回答1")
        self.ctx.add_user_message("问题2")
        assert self.ctx.get_turn_count() == 2
    
    def test_max_turns_compression(self):
        """测试超过最大轮次时自动压缩"""
        ctx = ContextManager(max_turns=3, compression_threshold=10000)
        ctx.set_system_prompt("你是助手")
        
        # 添加 4 轮对话（8 条消息）
        for i in range(4):
            ctx.add_user_message(f"问题{i}")
            ctx.add_assistant_message(f"回答{i}")
        
        # 超过 3 轮后应该触发压缩，只保留最近 3 轮（6 条）
        assert len(ctx.messages) <= 6
    
    # ============ 上下文压缩测试 ============
    
    def test_context_compression(self):
        """测试上下文过长时的压缩"""
        ctx = ContextManager(
            max_turns=100,
            compression_threshold=50,  # 设置很低以触发压缩
            max_context_tokens=100
        )
        ctx.set_system_prompt("你是助手")
        
        # 添加足够多的消息来触发压缩
        for i in range(10):
            ctx.add_user_message(f"这是一段较长的用户消息内容，编号为 {i}，包含一些详细信息")
            ctx.add_assistant_message(f"这是一段较长的助手回复内容，编号为 {i}，包含详细的解答")
        
        messages = ctx.get_messages()
        # 压缩后应该比原始消息少
        # 原始：10 * 2 = 20 条用户+助手消息
        # 压缩后：1 条摘要 + 6 条最近消息 = 7 条
        assert len(messages) < 21  # 20 messages + 1 system
    
    def test_compression_preserves_recent(self):
        """测试压缩保留最近的对话"""
        ctx = ContextManager(
            max_turns=100,
            compression_threshold=50,
            max_context_tokens=100
        )
        ctx.set_system_prompt("你是助手")
        
        # 添加多轮对话
        for i in range(8):
            ctx.add_user_message(f"消息{i}")
            ctx.add_assistant_message(f"回复{i}")
        
        messages = ctx.get_messages()
        # 最近的消息应该被保留
        last_message = messages[-1]
        assert last_message["role"] == "assistant"
    
    # ============ 追问测试 ============
    
    def test_follow_up_pure_conversation(self):
        """测试纯对话追问 - 上下文保持连续"""
        self.ctx.add_user_message("Python 是什么？")
        self.ctx.add_assistant_message("Python 是一种高级编程语言。")
        
        # 追问
        self.ctx.add_user_message("它有什么优点？")
        
        messages = self.ctx.get_messages()
        
        # 追问应该能看到之前的对话
        assert len(messages) == 4  # system + 2 user + 1 assistant
        assert messages[1]["content"] == "Python 是什么？"
        assert messages[2]["content"] == "Python 是一种高级编程语言。"
        assert messages[3]["content"] == "它有什么优点？"
    
    def test_follow_up_with_tools(self):
        """测试带工具的追问 - 工具结果保留在上下文中"""
        # 第一轮：查天气
        self.ctx.add_user_message("北京天气怎么样？")
        tool_calls = [{
            "id": "call_001",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city": "北京"}'}
        }]
        self.ctx.add_assistant_message(content="", tool_calls=tool_calls)
        self.ctx.add_tool_message("call_001", "weather", {
            "success": True, "city": "北京", "temperature": "22°C"
        })
        self.ctx.add_assistant_message("北京今天22°C，天气晴朗。")
        
        # 追问：带工具的追问
        self.ctx.add_user_message("那上海呢？")
        tool_calls2 = [{
            "id": "call_002",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city": "上海"}'}
        }]
        self.ctx.add_assistant_message(content="", tool_calls=tool_calls2)
        self.ctx.add_tool_message("call_002", "weather", {
            "success": True, "city": "上海", "temperature": "26°C"
        })
        
        messages = self.ctx.get_messages()
        
        # 上下文中应该包含两轮的完整信息
        assert len(messages) == 8  # system + 6条消息
        
        # 验证第一轮的天气数据还在
        assert any("北京" in m.get("content", "") for m in messages)
    
    # ============ 清空测试 ============
    
    def test_clear(self):
        self.ctx.add_user_message("你好")
        self.ctx.add_assistant_message("你好！")
        
        self.ctx.clear()
        
        assert len(self.ctx.messages) == 0
    
    # ============ Token 估算测试 ============
    
    def test_estimate_tokens(self):
        self.ctx.add_user_message("这是一段测试文本")
        tokens = self.ctx._estimate_tokens()
        assert tokens > 0
