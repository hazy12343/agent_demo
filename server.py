"""
FastAPI 服务端入口
提供 REST API 接口
"""
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import AgentRuntime
from session import SessionManager
from logger import trace_logger, setup_logging
from config import config

# 配置日志
setup_logging(config.log_level, config.log_file)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mini Agent API", version="1.0.0")

# 全局 Agent 运行时
agent = AgentRuntime()


# ============ 请求/响应模型 ============

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    user_id: str = Field(default="default", description="用户 ID")
    session_id: Optional[str] = Field(default=None, description="会话 ID（不传则创建新会话）")


class ChatResponse(BaseModel):
    response: str
    session_id: str
    trace_id: str
    tool_calls_used: list
    turns: int
    thinking: Optional[str] = None


class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    created_at: float
    last_active: float
    turns: int
    messages_count: int


# ============ API 路由 ============

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与 Agent 对话
    
    - 首次对话不传 session_id，系统会自动创建
    - 后续对话传入同一个 session_id 可以继续对话
    - 同一个 user_id 可以创建多个独立的 session
    """
    try:
        result = agent.chat(
            user_input=request.message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat 接口异常: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    """列出用户的所有会话"""
    sessions = agent.session_manager.list_user_sessions(user_id)
    return {"sessions": [
        {
            "session_id": s.session_id,
            "created_at": s.created_at,
            "last_active": s.last_active
        }
        for s in sessions
    ]}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """获取会话详情"""
    info = agent.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    return info


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    agent.clear_session_context(session_id)
    deleted = agent.session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {"message": "会话已删除"}


@app.get("/traces")
async def list_traces():
    """获取所有执行追踪记录"""
    traces = trace_logger.get_all_traces()
    return {"traces": [t.to_dict() for t in traces[-50:]]}


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    """获取单条追踪记录"""
    trace = trace_logger.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="追踪记录不存在")
    return trace.to_dict()


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "active_sessions": agent.session_manager.get_session_count(),
        "available_tools": agent.session_manager is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
