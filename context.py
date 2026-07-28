"""
Context 管理模块
负责管理对话上下文，包括消息历史、压缩策略等
"""
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from config import config

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """消息"""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    metadata: Optional[Dict[str, Any]] = None


class ContextManager:
    """
    Context 管理器
    
    功能：
    - 维护对话消息历史
    - 上下文长度控制与压缩
    - 系统提示管理
    """
    
    def __init__(
        self,
        max_turns: int = None,
        compression_threshold: int = None,
        max_context_tokens: int = None
    ):
        self.max_turns = max_turns or config.max_turns
        self.compression_threshold = compression_threshold or config.compression_threshold
        self.max_context_tokens = max_context_tokens or config.max_context_tokens
        self.messages: List[Message] = []
        self._system_prompt = ""
        self._tool_result_cache: Dict[str, Any] = {}
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self._system_prompt = prompt
    
    def add_user_message(self, content: str) -> Message:
        """添加用户消息"""
        msg = Message(role="user", content=content)
        self.messages.append(msg)
        self._check_limits()
        return msg
    
    def add_assistant_message(
        self,
        content: Optional[str],
        tool_calls: Optional[List[Dict]] = None,
        thinking: Optional[str] = None
    ) -> Message:
        """添加助手消息"""
        metadata = {}
        if thinking:
            metadata["thinking"] = thinking
        
        msg = Message(
            role="assistant",
            content=content or "",
            tool_calls=tool_calls,
            metadata=metadata
        )
        self.messages.append(msg)
        self._check_limits()
        return msg
    
    def add_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Message:
        """添加工具执行结果消息"""
        content = json.dumps(result, ensure_ascii=False)
        msg = Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            metadata={"tool_name": tool_name}
        )
        self.messages.append(msg)
        self._check_limits()
        return msg
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        获取消息列表（OpenAI API 格式）
        
        构建策略：
        1. 系统提示（始终在最前面）
        2. 压缩后的历史消息
        3. 最近的对话消息
        """
        result = []
        
        # 系统提示
        if self._system_prompt:
            result.append({
                "role": "system",
                "content": self._system_prompt
            })
        
        # 对话消息
        for msg in self.messages:
            m = {"role": msg.role, "content": msg.content}
            
            if msg.tool_call_id:
                m["tool_call_id"] = msg.tool_call_id
            
            if msg.tool_calls:
                m["tool_calls"] = msg.tool_calls
            
            result.append(m)
        
        return result
    
    def _estimate_tokens(self) -> int:
        """估算 token 数量（简单按字符数 / 4 估算）"""
        total = len(self._system_prompt)
        for msg in self.messages:
            total += len(msg.content)
            if msg.tool_calls:
                total += len(json.dumps(msg.tool_calls))
        return total // 4
    
    def _check_limits(self):
        """检查并执行压缩"""
        # 检查轮次限制（每2条消息 = 1轮：user + assistant）
        if len(self.messages) > self.max_turns * 2:
            self._compress_by_turns()
        
        # 检查 token 限制
        estimated_tokens = self._estimate_tokens()
        if estimated_tokens > self.compression_threshold:
            self._compress_context()
    
    def _compress_by_turns(self):
        """按轮次压缩：保留最近的 N 轮对话"""
        keep_count = self.max_turns * 2
        if len(self.messages) > keep_count:
            removed_count = len(self.messages) - keep_count
            self.messages = self.messages[-keep_count:]
            logger.info(f"轮次压缩: 移除了 {removed_count} 条消息")
    
    def _compress_context(self):
        """
        基础上下文压缩
        
        策略：
        1. 保留系统提示
        2. 保留最近 3 轮对话
        3. 中间部分压缩为摘要
        """
        if len(self.messages) <= 6:
            return
        
        # 保留最近 6 条消息（3轮）
        recent = self.messages[-6:]
        old = self.messages[:-6]
        
        # 将旧消息压缩为摘要
        summary_parts = []
        for msg in old:
            if msg.role == "user":
                summary_parts.append(f"用户问过: {msg.content[:100]}")
            elif msg.role == "assistant" and msg.content:
                summary_parts.append(f"助手回答: {msg.content[:100]}")
            elif msg.role == "tool" and msg.metadata:
                summary_parts.append(f"使用了工具: {msg.metadata.get('tool_name', 'unknown')}")
        
        if summary_parts:
            summary = "\n".join(summary_parts)
            compressed_msg = Message(
                role="system",
                content=f"[历史对话摘要]\n{summary}\n[摘要结束]"
            )
            self.messages = [compressed_msg] + recent
            logger.info(f"上下文压缩: 将 {len(old)} 条消息压缩为摘要")
    
    def get_turn_count(self) -> int:
        """获取当前对话轮次数"""
        user_messages = sum(1 for m in self.messages if m.role == "user")
        return user_messages
    
    def clear(self):
        """清空上下文"""
        self.messages.clear()
        self._tool_result_cache.clear()
