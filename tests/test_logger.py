"""
测试日志追踪模块
"""
import pytest
from logger import TraceLogger, ExecutionTrace, TraceEntry, setup_logging


class TestTraceLogger:
    """追踪日志器测试"""
    
    def setup_method(self):
        self.trace_logger = TraceLogger()
    
    def test_start_trace(self):
        trace = self.trace_logger.start_trace("session_1", "你好")
        
        assert trace.trace_id is not None
        assert trace.session_id == "session_1"
        assert trace.user_input == "你好"
        assert len(trace.entries) == 1  # user_input entry
    
    def test_log_llm_call(self):
        trace = self.trace_logger.start_trace("session_1", "test")
        
        self.trace_logger.log_llm_call(
            trace,
            messages_count=5,
            response_data={
                "finish_reason": "stop",
                "tool_calls": None,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            },
            duration_ms=500.0
        )
        
        llm_entries = [e for e in trace.entries if e.event_type == "llm_call"]
        assert len(llm_entries) == 1
        assert llm_entries[0].duration_ms == 500.0
        assert llm_entries[0].data["messages_count"] == 5
    
    def test_log_tool_call(self):
        trace = self.trace_logger.start_trace("session_1", "test")
        
        self.trace_logger.log_tool_call(
            trace,
            tool_name="calculator",
            arguments={"expression": "2+3"},
            result={"success": True, "result": 5},
            duration_ms=10.0
        )
        
        tool_entries = [e for e in trace.entries if e.event_type == "tool_call"]
        assert len(tool_entries) == 1
        assert tool_entries[0].data["tool_name"] == "calculator"
        assert tool_entries[0].data["success"] is True
    
    def test_log_agent_response(self):
        trace = self.trace_logger.start_trace("session_1", "test")
        
        self.trace_logger.log_agent_response(trace, "这是最终回复")
        
        assert trace.final_response == "这是最终回复"
        assert trace.completed_at is not None
    
    def test_get_trace(self):
        trace = self.trace_logger.start_trace("session_1", "test")
        retrieved = self.trace_logger.get_trace(trace.trace_id)
        assert retrieved is trace
    
    def test_get_all_traces(self):
        self.trace_logger.start_trace("s1", "input1")
        self.trace_logger.start_trace("s2", "input2")
        
        traces = self.trace_logger.get_all_traces()
        assert len(traces) == 2
    
    def test_clear_traces(self):
        self.trace_logger.start_trace("s1", "input1")
        self.trace_logger.clear_traces()
        
        assert len(self.trace_logger.get_all_traces()) == 0


class TestExecutionTrace:
    """执行追踪数据类测试"""
    
    def test_total_duration(self):
        trace = ExecutionTrace(
            trace_id="test",
            session_id="s1",
            user_input="test"
        )
        trace.complete("done")
        
        assert trace.total_duration_ms >= 0
    
    def test_to_dict(self):
        trace = ExecutionTrace(
            trace_id="test",
            session_id="s1",
            user_input="test input"
        )
        trace.add_entry("user_input", {"message": "test"})
        trace.complete("done")
        
        d = trace.to_dict()
        assert d["trace_id"] == "test"
        assert d["session_id"] == "s1"
        assert d["user_input"] == "test input"
        assert d["final_response"] == "done"
        assert len(d["entries"]) == 1
    
    def test_add_entry_with_error(self):
        trace = ExecutionTrace(
            trace_id="test",
            session_id="s1",
            user_input="test"
        )
        trace.add_entry("error", {"detail": "something went wrong"}, error="Exception")
        
        assert trace.entries[0].error == "Exception"
