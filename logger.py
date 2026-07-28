"""
日志追踪模块
提供工具调用 trace、执行日志等功能
"""
import logging
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class TraceEntry:
    """单条追踪记录"""
    timestamp: float
    event_type: str  # "llm_call", "tool_call", "tool_result", "user_input", "agent_response"
    data: Dict[str, Any]
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ExecutionTrace:
    """执行追踪（完整的一次 agent 执行链路）"""
    trace_id: str
    session_id: str
    user_input: str
    entries: List[TraceEntry] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    final_response: Optional[str] = None
    
    def add_entry(self, event_type: str, data: Dict[str, Any],
                  duration_ms: float = None, error: str = None):
        self.entries.append(TraceEntry(
            timestamp=time.time(),
            event_type=event_type,
            data=data,
            duration_ms=duration_ms,
            error=error
        ))
    
    def complete(self, final_response: str):
        self.completed_at = time.time()
        self.final_response = final_response
    
    @property
    def total_duration_ms(self) -> float:
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "total_duration_ms": self.total_duration_ms,
            "steps_count": len(self.entries),
            "final_response": self.final_response,
            "entries": [asdict(e) for e in self.entries]
        }


class TraceLogger:
    """追踪日志管理器"""
    
    def __init__(self):
        self._traces: Dict[str, ExecutionTrace] = {}
        self._logger = logging.getLogger("agent_trace")
    
    def start_trace(self, session_id: str, user_input: str) -> ExecutionTrace:
        """开始一次新的追踪"""
        trace_id = str(uuid.uuid4())[:8]
        trace = ExecutionTrace(
            trace_id=trace_id,
            session_id=session_id,
            user_input=user_input
        )
        self._traces[trace_id] = trace
        
        trace.add_entry("user_input", {"message": user_input})
        self._logger.info(f"[Trace {trace_id}] 用户输入: {user_input[:100]}")
        
        return trace
    
    def log_llm_call(self, trace: ExecutionTrace, messages_count: int,
                     response_data: Dict, duration_ms: float):
        """记录 LLM 调用"""
        trace.add_entry(
            "llm_call",
            {
                "messages_count": messages_count,
                "finish_reason": response_data.get("finish_reason"),
                "has_tool_calls": response_data.get("tool_calls") is not None,
                "usage": response_data.get("usage", {})
            },
            duration_ms=duration_ms
        )
        self._logger.info(
            f"[Trace {trace.trace_id}] LLM 调用: "
            f"messages={messages_count}, duration={duration_ms:.0f}ms"
        )
    
    def log_tool_call(self, trace: ExecutionTrace, tool_name: str,
                      arguments: Dict, result: Dict, duration_ms: float,
                      error: str = None):
        """记录工具调用"""
        trace.add_entry(
            "tool_call",
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "result_summary": str(result)[:200],
                "success": result.get("success", True)
            },
            duration_ms=duration_ms,
            error=error
        )
        self._logger.info(
            f"[Trace {trace.trace_id}] 工具调用: {tool_name}({list(arguments.keys())}), "
            f"duration={duration_ms:.0f}ms, success={result.get('success', True)}"
        )
    
    def log_agent_response(self, trace: ExecutionTrace, response: str):
        """记录 Agent 最终响应"""
        trace.add_entry("agent_response", {"response": response[:500]})
        trace.complete(response)
        self._logger.info(
            f"[Trace {trace.trace_id}] Agent 响应: {response[:100]}... "
            f"(total: {trace.total_duration_ms:.0f}ms, steps: {len(trace.entries)})"
        )
    
    def get_trace(self, trace_id: str) -> Optional[ExecutionTrace]:
        """获取追踪记录"""
        return self._traces.get(trace_id)
    
    def get_all_traces(self) -> List[ExecutionTrace]:
        """获取所有追踪记录"""
        return list(self._traces.values())
    
    def clear_traces(self):
        """清空追踪记录"""
        self._traces.clear()


def setup_logging(level: str = "INFO", log_file: str = None):
    """配置日志"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件 handler（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger


# 全局追踪日志器
trace_logger = TraceLogger()
