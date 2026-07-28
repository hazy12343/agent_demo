"""
Session 管理模块
支持多用户多会话的独立管理
"""
import time
import uuid
import logging
from typing import Dict, Optional
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """会话数据"""
    session_id: str
    user_id: str
    created_at: float
    last_active: float
    messages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class SessionManager:
    """
    Session 管理器
    
    支持：
    - 多用户多会话
    - 会话超时清理
    - 线程安全
    """
    
    def __init__(self, timeout: int = 3600):
        self._sessions: Dict[str, Session] = {}
        self._timeout = timeout
        self._lock = Lock()
    
    def create_session(self, user_id: str, session_id: Optional[str] = None) -> Session:
        """创建新会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        now = time.time()
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_active=now
        )
        
        with self._lock:
            self._sessions[session_id] = session
        
        logger.info(f"创建会话: user={user_id}, session={session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session:
                # 检查是否超时
                if time.time() - session.last_active > self._timeout:
                    logger.info(f"会话超时: {session_id}")
                    del self._sessions[session_id]
                    return None
                
                # 更新活跃时间
                session.last_active = time.time()
            
            return session
    
    def get_or_create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Session:
        """获取或创建会话"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        
        return self.create_session(user_id, session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"删除会话: {session_id}")
                return True
            return False
    
    def list_user_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        with self._lock:
            return [
                s for s in self._sessions.values()
                if s.user_id == user_id
            ]
    
    def cleanup_expired(self):
        """清理过期会话"""
        now = time.time()
        expired = []
        
        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_active > self._timeout:
                    expired.append(session_id)
            
            for session_id in expired:
                del self._sessions[session_id]
        
        if expired:
            logger.info(f"清理了 {len(expired)} 个过期会话")
        
        return len(expired)
    
    def get_session_count(self) -> int:
        """获取活跃会话数"""
        with self._lock:
            return len(self._sessions)
