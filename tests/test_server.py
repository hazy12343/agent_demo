"""测试 FastAPI 服务端
使用 Mock Agent 来测试 API 端点
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from server import app, agent


class TestServerAPI:
    """API 端点测试"""
    
    def setup_method(self):
        self.client = TestClient(app)
        # 清理状态
        agent.session_manager._sessions.clear()
        agent._contexts.clear()
    
    def _mock_agent_chat(self, return_value=None):
        """Mock agent.chat 方法"""
        if return_value is None:
            return_value = {
                "response": "你好！有什么可以帮助你的吗？",
                "session_id": "test_session_123",
                "trace_id": "trace_456",
                "tool_calls_used": [],
                "turns": 1,
                "thinking": None
            }
        return patch.object(agent, "chat", return_value=return_value)
    
    # ============ 健康检查 ============
    
    def test_health_check(self):
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    # ============ Chat 端点测试 ============
    
    def test_chat_basic(self):
        """测试基本对话"""
        with self._mock_agent_chat():
            response = self.client.post("/chat", json={
                "message": "你好",
                "user_id": "user_a"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "session_id" in data
        assert "trace_id" in data
        assert data["turns"] == 1
    
    def test_chat_with_session_id(self):
        """测试带 session_id 的对话"""
        with self._mock_agent_chat():
            response = self.client.post("/chat", json={
                "message": "继续对话",
                "user_id": "user_a",
                "session_id": "existing_session"
            })
        
        assert response.status_code == 200
    
    def test_chat_with_tool_calls(self):
        """测试包含工具调用的对话"""
        tool_result = {
            "response": "北京今天22°C，天气晴朗。",
            "session_id": "session_weather",
            "trace_id": "trace_weather",
            "tool_calls_used": [
                {
                    "tool": "weather",
                    "arguments": {"city": "北京"},
                    "result": {"success": True, "temperature": "22°C"}
                }
            ],
            "turns": 1,
            "thinking": None
        }
        
        with self._mock_agent_chat(tool_result):
            response = self.client.post("/chat", json={
                "message": "北京天气怎么样？",
                "user_id": "user_a"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["tool_calls_used"]) == 1
        assert data["tool_calls_used"][0]["tool"] == "weather"
    
    def test_chat_missing_message(self):
        """测试缺少消息字段"""
        response = self.client.post("/chat", json={
            "user_id": "user_a"
        })
        assert response.status_code == 422  # Pydantic 验证错误
    
    # ============ Session 端点测试 ============
    
    def test_list_sessions(self):
        """测试列出用户会话"""
        # 先创建一些会话
        agent.session_manager.create_session("user_a")
        agent.session_manager.create_session("user_a")
        
        response = self.client.get("/sessions/user_a")
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 2
    
    def test_get_session(self):
        """测试获取会话详情"""
        session = agent.session_manager.create_session("user_a")
        agent._contexts[session.session_id] = MagicMock(
            get_turn_count=lambda: 3,
            messages=[1, 2, 3, 4, 5, 6]
        )
        
        response = self.client.get(f"/session/{session.session_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session.session_id
        assert data["turns"] == 3
    
    def test_get_nonexistent_session(self):
        """测试获取不存在的会话"""
        response = self.client.get("/session/nonexistent")
        assert response.status_code == 404
    
    def test_delete_session(self):
        """测试删除会话"""
        session = agent.session_manager.create_session("user_a")
        
        response = self.client.delete(f"/session/{session.session_id}")
        assert response.status_code == 200
        
        # 验证已删除
        assert agent.session_manager.get_session(session.session_id) is None
    
    # ============ Trace 端点测试 ============
    
    def test_list_traces(self):
        """测试列出追踪记录"""
        response = self.client.get("/traces")
        assert response.status_code == 200
        data = response.json()
        assert "traces" in data
    
    def test_get_nonexistent_trace(self):
        """测试获取不存在的追踪记录"""
        response = self.client.get("/traces/nonexistent")
        assert response.status_code == 404
    
    # ============ 多用户多会话集成测试 ============
    
    def test_multi_user_multi_session(self):
        """测试多用户多会话场景"""
        with self._mock_agent_chat():
            # 用户 A 窗口 1
            r1 = self.client.post("/chat", json={
                "message": "查天气",
                "user_id": "user_a"
            })
            s1_id = r1.json()["session_id"]
            
            # 用户 A 窗口 2
            r2 = self.client.post("/chat", json={
                "message": "写周报",
                "user_id": "user_a"
            })
            s2_id = r2.json()["session_id"]
            
            # 用户 B
            r3 = self.client.post("/chat", json={
                "message": "你好",
                "user_id": "user_b"
            })
        
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200
