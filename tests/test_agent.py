"""
测试 Agent 核心运行时
使用 Mock LLM 来测试 Agent 循环逻辑，无需真实 API 调用
"""
import pytest
from unittest.mock import MagicMock, patch
from agent import AgentRuntime
from session import SessionManager
from context import ContextManager
from logger import trace_logger


class MockLLMClient:
    """Mock LLM 客户端，模拟不同的 LLM 响应场景"""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
    
    def chat(self, messages, tools=None, temperature=0.7, max_tokens=2000):
        """模拟 LLM 调用"""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            # 默认回复
            response = {
                "content": "这是默认回复。",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
        self.call_count += 1
        return response


# ============ 预定义的 LLM 响应 ============

DIRECT_RESPONSE = {
    "content": "你好！我是一个智能助手，有什么可以帮助你的吗？",
    "tool_calls": None,
    "finish_reason": "stop",
    "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
}

TOOL_CALL_WEATHER = {
    "content": None,
    "tool_calls": [
        {
            "id": "call_001",
            "name": "weather",
            "arguments": {"city": "北京"}
        }
    ],
    "finish_reason": "tool_calls",
    "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50}
}

TOOL_CALL_CALCULATOR = {
    "content": None,
    "tool_calls": [
        {
            "id": "call_002",
            "name": "calculator",
            "arguments": {"expression": "100 * 1.05"}
        }
    ],
    "finish_reason": "tool_calls",
    "usage": {"prompt_tokens": 35, "completion_tokens": 20, "total_tokens": 55}
}

TOOL_CALL_SEARCH = {
    "content": None,
    "tool_calls": [
        {
            "id": "call_003",
            "name": "search",
            "arguments": {"query": "python 教程"}
        }
    ],
    "finish_reason": "tool_calls",
    "usage": {"prompt_tokens": 25, "completion_tokens": 18, "total_tokens": 43}
}

FOLLOW_UP_RESPONSE = {
    "content": "北京今天22°C，天气晴朗，适合外出。你还需要什么帮助吗？",
    "tool_calls": None,
    "finish_reason": "stop",
    "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}
}

MULTI_TOOL_CALL = {
    "content": "让我先查天气再算一下。",
    "tool_calls": [
        {
            "id": "call_010",
            "name": "weather",
            "arguments": {"city": "上海"}
        }
    ],
    "finish_reason": "tool_calls",
    "usage": {"prompt_tokens": 40, "completion_tokens": 22, "total_tokens": 62}
}


class TestAgentRuntime:
    """Agent 运行时测试"""
    
    def setup_method(self):
        """每个测试前创建新的 agent"""
        self.session_manager = SessionManager(timeout=3600)
    
    def _create_agent(self, llm_responses):
        """创建带有 Mock LLM 的 Agent"""
        mock_llm = MockLLMClient(llm_responses)
        agent = AgentRuntime(
            llm_client=mock_llm,
            session_manager=self.session_manager
        )
        return agent
    
    # ============ 直接回复测试 ============
    
    def test_direct_response(self):
        """测试不需要工具的简单对话"""
        agent = self._create_agent([DIRECT_RESPONSE])
        
        result = agent.chat("你好", user_id="user_a")
        
        assert result["response"] == "你好！我是一个智能助手，有什么可以帮助你的吗？"
        assert result["session_id"] is not None
        assert result["trace_id"] is not None
        assert result["tool_calls_used"] == []
        assert result["turns"] == 1
    
    # ============ 工具调用测试 ============
    
    def test_weather_tool_call(self):
        """测试天气工具调用"""
        agent = self._create_agent([TOOL_CALL_WEATHER, FOLLOW_UP_RESPONSE])
        
        result = agent.chat("北京天气怎么样？", user_id="user_a")
        
        assert len(result["tool_calls_used"]) == 1
        assert result["tool_calls_used"][0]["tool"] == "weather"
        assert result["tool_calls_used"][0]["result"]["success"] is True
        assert "北京" in result["response"]
    
    def test_calculator_tool_call(self):
        """测试计算器工具调用"""
        after_calc_response = {
            "content": "100 * 1.05 = 105.0，所以答案是105。",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 40, "completion_tokens": 15, "total_tokens": 55}
        }
        agent = self._create_agent([TOOL_CALL_CALCULATOR, after_calc_response])
        
        result = agent.chat("100 的 5% 利息是多少？", user_id="user_a")
        
        assert len(result["tool_calls_used"]) == 1
        assert result["tool_calls_used"][0]["tool"] == "calculator"
        assert result["tool_calls_used"][0]["result"]["result"] == 105.0
    
    def test_search_tool_call(self):
        """测试搜索工具调用"""
        after_search_response = {
            "content": "我找到了一些 Python 教程，包括官方文档和入门教程。",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 45, "completion_tokens": 20, "total_tokens": 65}
        }
        agent = self._create_agent([TOOL_CALL_SEARCH, after_search_response])
        
        result = agent.chat("帮我搜索 Python 教程", user_id="user_a")
        
        assert len(result["tool_calls_used"]) == 1
        assert result["tool_calls_used"][0]["tool"] == "search"
    
    # ============ Session 管理测试 ============
    
    def test_session_creation(self):
        """测试自动创建 session"""
        agent = self._create_agent([DIRECT_RESPONSE])
        
        result = agent.chat("你好", user_id="user_a")
        
        assert result["session_id"] is not None
        assert self.session_manager.get_session_count() == 1
    
    def test_session_reuse(self):
        """测试复用已有 session"""
        agent = self._create_agent([DIRECT_RESPONSE, DIRECT_RESPONSE])
        
        result1 = agent.chat("第一条消息", user_id="user_a")
        session_id = result1["session_id"]
        
        result2 = agent.chat("第二条消息", user_id="user_a", session_id=session_id)
        
        assert result2["session_id"] == session_id
        assert result2["turns"] == 2
    
    def test_independent_sessions(self):
        """测试同一用户的独立会话（多个窗口）"""
        agent = self._create_agent([
            TOOL_CALL_WEATHER, FOLLOW_UP_RESPONSE,
            DIRECT_RESPONSE
        ])
        
        # 窗口1：查天气
        result1 = agent.chat("北京天气怎么样？", user_id="user_a")
        window1_id = result1["session_id"]
        
        # 窗口2：写周报
        result2 = agent.chat("帮我写周报", user_id="user_a")
        window2_id = result2["session_id"]
        
        # 两个窗口的 session_id 应该不同
        assert window1_id != window2_id
        
        # 窗口1应该有工具调用，窗口2没有
        assert len(result1["tool_calls_used"]) == 1
        assert len(result2["tool_calls_used"]) == 0
    
    # ============ 追问测试 ============
    
    def test_follow_up_conversation(self):
        """测试追问功能 - 基于之前的上下文"""
        agent = self._create_agent([
            TOOL_CALL_WEATHER,
            FOLLOW_UP_RESPONSE,
            {
                "content": "根据之前查询的结果，北京今天22°C，晴天，非常适合外出活动。",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 60, "completion_tokens": 20, "total_tokens": 80}
            }
        ])
        
        # 第一轮：查天气
        result1 = agent.chat("北京天气怎么样？", user_id="user_a")
        session_id = result1["session_id"]
        
        # 追问
        result2 = agent.chat("适合外出吗？", user_id="user_a", session_id=session_id)
        
        assert result2["turns"] == 2
        assert "22" in result2["response"] or "晴" in result2["response"]
    
    def test_follow_up_with_tool(self):
        """测试带工具的追问"""
        agent = self._create_agent([
            TOOL_CALL_WEATHER,
            FOLLOW_UP_RESPONSE,
            # 追问：换城市查天气
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_005",
                        "name": "weather",
                        "arguments": {"city": "上海"}
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 70, "completion_tokens": 20, "total_tokens": 90}
            },
            {
                "content": "上海今天26°C，多云天气。",
                "tool_calls": None,
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 80, "completion_tokens": 15, "total_tokens": 95}
            }
        ])
        
        # 第一轮
        result1 = agent.chat("北京天气怎么样？", user_id="user_a")
        session_id = result1["session_id"]
        
        # 追问（带工具）
        result2 = agent.chat("那上海呢？", user_id="user_a", session_id=session_id)
        
        assert result2["turns"] == 2
        assert len(result2["tool_calls_used"]) == 1
        assert result2["tool_calls_used"][0]["tool"] == "weather"
    
    # ============ 异常处理测试 ============
    
    def test_tool_not_found(self):
        """测试调用不存在的工具"""
        bad_tool_response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_bad",
                    "name": "nonexistent_tool",
                    "arguments": {"param": "value"}
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
        }
        after_error = {
            "content": "抱歉，该工具不可用。",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
        }
        agent = self._create_agent([bad_tool_response, after_error])
        
        result = agent.chat("使用不存在的工具", user_id="user_a")
        
        # 工具调用应该失败但不会崩溃
        assert len(result["tool_calls_used"]) == 1
        assert result["tool_calls_used"][0]["result"]["success"] is False
    
    def test_max_iterations_reached(self):
        """测试达到最大迭代次数"""
        # 每次都返回工具调用
        infinite_tool_calls = [TOOL_CALL_WEATHER] * 10
        agent = self._create_agent(infinite_tool_calls)
        agent.max_iterations = 3
        
        result = agent.chat("测试最大迭代", user_id="user_a")
        
        assert "尝试了多次" in result["response"]
    
    def test_llm_exception_handling(self):
        """测试 LLM 调用异常"""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = Exception("API 调用失败")
        
        agent = AgentRuntime(
            llm_client=mock_llm,
            session_manager=self.session_manager
        )
        
        result = agent.chat("测试异常", user_id="user_a")
        
        assert "错误" in result["response"]
    
    # ============ 追踪日志测试 ============
    
    def test_trace_logging(self):
        """测试执行追踪记录"""
        agent = self._create_agent([TOOL_CALL_WEATHER, FOLLOW_UP_RESPONSE])
        
        trace_logger.clear_traces()
        result = agent.chat("北京天气怎么样？", user_id="user_a")
        
        trace_id = result["trace_id"]
        trace = trace_logger.get_trace(trace_id)
        
        assert trace is not None
        assert trace.user_input == "北京天气怎么样？"
        assert trace.final_response is not None
        assert len(trace.entries) >= 3  # user_input + llm_call + tool_call + llm_call + agent_response
        assert trace.total_duration_ms >= 0
    
    def test_trace_tool_details(self):
        """测试追踪记录的工具调用详情"""
        agent = self._create_agent([TOOL_CALL_CALCULATOR, {
            "content": "结果是 105.0",
            "tool_calls": None,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
        }])
        
        trace_logger.clear_traces()
        result = agent.chat("计算 100 * 1.05", user_id="user_a")
        
        trace = trace_logger.get_trace(result["trace_id"])
        
        # 查找 tool_call 类型的条目
        tool_entries = [e for e in trace.entries if e.event_type == "tool_call"]
        assert len(tool_entries) == 1
        assert tool_entries[0].data["tool_name"] == "calculator"
        assert tool_entries[0].data["success"] is True
    
    # ============ Session 信息测试 ============
    
    def test_get_session_info(self):
        """测试获取会话信息"""
        agent = self._create_agent([DIRECT_RESPONSE])
        
        result = agent.chat("你好", user_id="user_a")
        
        info = agent.get_session_info(result["session_id"])
        assert info is not None
        assert info["user_id"] == "user_a"
        assert info["turns"] == 1
    
    def test_clear_session_context(self):
        """测试清除会话上下文"""
        agent = self._create_agent([DIRECT_RESPONSE])
        
        result = agent.chat("你好", user_id="user_a")
        session_id = result["session_id"]
        
        agent.clear_session_context(session_id)
        
        # 清除后应该创建新的 context
        assert session_id not in agent._contexts
