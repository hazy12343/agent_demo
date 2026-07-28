"""
测试 Session 管理模块
"""
import time
import pytest
from session import SessionManager, Session


class TestSessionManager:
    """Session 管理器测试"""
    
    def setup_method(self):
        self.manager = SessionManager(timeout=5)  # 5秒超时，方便测试
    
    # ============ 创建会话测试 ============
    
    def test_create_session(self):
        session = self.manager.create_session("user_a")
        assert session.user_id == "user_a"
        assert session.session_id is not None
        assert len(session.session_id) > 0
    
    def test_create_session_with_custom_id(self):
        session = self.manager.create_session("user_a", session_id="custom_123")
        assert session.session_id == "custom_123"
    
    def test_create_multiple_sessions_same_user(self):
        """同一用户可以创建多个独立会话（模拟多个窗口）"""
        s1 = self.manager.create_session("user_a")
        s2 = self.manager.create_session("user_a")
        
        assert s1.session_id != s2.session_id
        
        sessions = self.manager.list_user_sessions("user_a")
        assert len(sessions) == 2
    
    # ============ 获取会话测试 ============
    
    def test_get_session(self):
        created = self.manager.create_session("user_a")
        retrieved = self.manager.get_session(created.session_id)
        
        assert retrieved is not None
        assert retrieved.session_id == created.session_id
    
    def test_get_nonexistent_session(self):
        result = self.manager.get_session("nonexistent")
        assert result is None
    
    def test_get_or_create_new(self):
        session = self.manager.get_or_create_session("user_a")
        assert session is not None
        assert session.user_id == "user_a"
    
    def test_get_or_create_existing(self):
        created = self.manager.create_session("user_a")
        retrieved = self.manager.get_or_create_session(
            "user_a", session_id=created.session_id
        )
        assert retrieved.session_id == created.session_id
    
    # ============ 会话独立性测试 ============
    
    def test_session_independence(self):
        """测试不同会话之间的独立性"""
        # 模拟用户 A 的两个窗口
        window1 = self.manager.create_session("user_a")
        window2 = self.manager.create_session("user_a")
        
        # 窗口1记录天气相关信息
        window1.messages.append({"role": "user", "content": "查天气"})
        window1.metadata["topic"] = "weather"
        
        # 窗口2记录周报相关信息
        window2.messages.append({"role": "user", "content": "写周报"})
        window2.metadata["topic"] = "report"
        
        # 验证两个窗口互不影响
        w1 = self.manager.get_session(window1.session_id)
        w2 = self.manager.get_session(window2.session_id)
        
        assert len(w1.messages) == 1
        assert w1.messages[0]["content"] == "查天气"
        assert w1.metadata["topic"] == "weather"
        
        assert len(w2.messages) == 1
        assert w2.messages[0]["content"] == "写周报"
        assert w2.metadata["topic"] == "report"
    
    def test_multiple_users_independence(self):
        """测试不同用户之间的独立性"""
        user_a_session = self.manager.create_session("user_a")
        user_b_session = self.manager.create_session("user_b")
        
        user_a_session.messages.append({"role": "user", "content": "用户A的消息"})
        
        a_sessions = self.manager.list_user_sessions("user_a")
        b_sessions = self.manager.list_user_sessions("user_b")
        
        assert len(a_sessions) == 1
        assert len(b_sessions) == 1
        assert len(a_sessions[0].messages) == 1
        assert len(b_sessions[0].messages) == 0
    
    # ============ 会话超时测试 ============
    
    def test_session_timeout(self):
        """测试会话超时"""
        manager = SessionManager(timeout=1)  # 1秒超时
        session = manager.create_session("user_a")
        
        # 会话应该存在
        assert manager.get_session(session.session_id) is not None
        
        # 等待超时
        time.sleep(1.5)
        
        # 会话应该已经过期
        assert manager.get_session(session.session_id) is None
    
    def test_session_activity_refresh(self):
        """测试活动会话的刷新"""
        manager = SessionManager(timeout=2)
        session = manager.create_session("user_a")
        
        # 在超时前访问，刷新活跃时间
        time.sleep(1)
        refreshed = manager.get_session(session.session_id)
        assert refreshed is not None
        
        # 再等1秒（总共2秒），但因为刷新了，所以不应该超时
        time.sleep(1)
        still_alive = manager.get_session(session.session_id)
        assert still_alive is not None
    
    # ============ 删除会话测试 ============
    
    def test_delete_session(self):
        session = self.manager.create_session("user_a")
        assert self.manager.delete_session(session.session_id) is True
        assert self.manager.get_session(session.session_id) is None
    
    def test_delete_nonexistent_session(self):
        assert self.manager.delete_session("nonexistent") is False
    
    # ============ 清理过期会话测试 ============
    
    def test_cleanup_expired(self):
        manager = SessionManager(timeout=1)
        manager.create_session("user_a")
        manager.create_session("user_b")
        
        time.sleep(1.5)
        
        cleaned = manager.cleanup_expired()
        assert cleaned == 2
        assert manager.get_session_count() == 0
    
    # ============ 会话计数测试 ============
    
    def test_session_count(self):
        self.manager.create_session("user_a")
        self.manager.create_session("user_b")
        self.manager.create_session("user_a")
        
        assert self.manager.get_session_count() == 3
    
    # ============ 继续对话测试 ============
    
    def test_resume_conversation(self):
        """测试用户可以随时接着之前的会话继续聊"""
        session = self.manager.create_session("user_a")
        
        # 第一轮对话
        session.messages.append({"role": "user", "content": "帮我查北京天气"})
        session.messages.append({"role": "assistant", "content": "北京今天22°C，晴天"})
        
        # 过一段时间后继续同一个会话
        retrieved = self.manager.get_or_create_session(
            "user_a", session_id=session.session_id
        )
        
        assert len(retrieved.messages) == 2
        assert retrieved.messages[0]["content"] == "帮我查北京天气"
        
        # 继续对话
        retrieved.messages.append({"role": "user", "content": "那明天呢？"})
        assert len(retrieved.messages) == 3
