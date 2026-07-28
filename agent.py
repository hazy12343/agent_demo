"""
Agent 核心运行时（Runtime）
实现 Agent 的基本循环：接收输入 → 判断调用工具/直接回复 → 执行工具 → 继续循环/返回结果
"""
import json
import time
import logging
from typing import Any, Dict, List, Optional

from llm_client import LLMClient, get_llm_client
from parser import OutputParser, ParsedOutput, ToolCall
from context import ContextManager
from session import Session, SessionManager
from tools import tool_registry, register_all_tools
from logger import trace_logger, ExecutionTrace
from config import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个智能助手 Agent。你可以使用以下工具来帮助用户解决问题：

1. calculator - 计算器：执行数学运算
2. search - 搜索：搜索信息
3. weather - 天气：查询城市天气

规则：
- 如果用户的问题需要查询信息或进行计算，请使用对应的工具
- 如果不需要工具，直接回复用户
- 工具返回结果后，请基于结果给出完整的回答
- 保持回复简洁、有帮助
"""


class AgentRuntime:
    """
    Agent 运行时
    
    核心循环：
    1. 接收用户输入
    2. 调用 LLM 判断是直接回复还是调用工具
    3. 如果需要调用工具，执行工具并获取结果
    4. 将工具结果反馈给 LLM，判断继续循环还是返回结果
    """
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        session_manager: Optional[SessionManager] = None,
        max_iterations: int = 5
    ):
        self.llm = llm_client or get_llm_client()
        self.session_manager = session_manager or SessionManager(
            timeout=config.session_timeout
        )
        self.parser = OutputParser()
        self.max_iterations = max_iterations  # 单次请求最大工具调用迭代次数
        
        # 注册所有工具
        register_all_tools()
        
        # 每个 session 对应一个 ContextManager
        self._contexts: Dict[str, ContextManager] = {}
    
    def _get_context(self, session_id: str) -> ContextManager:
        """获取或创建 Context 管理器"""
        if session_id not in self._contexts:
            ctx = ContextManager()
            ctx.set_system_prompt(SYSTEM_PROMPT)
            self._contexts[session_id] = ctx
        return self._contexts[session_id]
    
    def chat(
        self,
        user_input: str,
        user_id: str = "default",
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理用户输入，返回响应
        
        Returns:
            {
                "response": str,           # Agent 回复
                "session_id": str,         # 会话 ID
                "trace_id": str,           # 追踪 ID
                "tool_calls_used": list,   # 使用的工具列表
                "turns": int              # 当前对话轮次
            }
        """
        # 1. 获取/创建会话
        session = self.session_manager.get_or_create_session(user_id, session_id)
        context = self._get_context(session.session_id)
        
        # 2. 开始追踪
        trace = trace_logger.start_trace(session.session_id, user_input)
        
        # 3. 将用户输入加入上下文
        context.add_user_message(user_input)
        
        # 4. Agent 循环
        tool_calls_used = []
        final_response = ""
        iteration = 0
        
        try:
            while iteration < self.max_iterations:
                iteration += 1
                
                # 4a. 调用 LLM
                messages = context.get_messages()
                tools_schema = tool_registry.get_all_schemas()
                
                start_time = time.time()
                llm_response = self.llm.chat(
                    messages=messages,
                    tools=tools_schema if tools_schema else None
                )
                llm_duration = (time.time() - start_time) * 1000
                
                trace_logger.log_llm_call(
                    trace, len(messages), llm_response, llm_duration
                )
                
                # 4b. 解析 LLM 输出
                parsed = self.parser.parse(llm_response)
                
                # 4c. 判断：是否有工具调用？
                if parsed.tool_calls:
                    # 记录助手的 tool_calls 消息
                    raw_tool_calls = llm_response.get("tool_calls", [])
                    tool_call_msgs = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"], ensure_ascii=False)
                            }
                        }
                        for tc in raw_tool_calls
                    ]
                    context.add_assistant_message(
                        content=parsed.thinking or "",
                        tool_calls=tool_call_msgs,
                        thinking=parsed.thinking
                    )
                    
                    # 执行每个工具调用
                    for tool_call in parsed.tool_calls:
                        tool_start = time.time()
                        tool_result = self._execute_tool(tool_call)
                        tool_duration = (time.time() - tool_start) * 1000
                        
                        tool_calls_used.append({
                            "tool": tool_call.name,
                            "arguments": tool_call.arguments,
                            "result": tool_result
                        })
                        
                        trace_logger.log_tool_call(
                            trace,
                            tool_call.name,
                            tool_call.arguments,
                            tool_result,
                            tool_duration
                        )
                        
                        # 将工具结果加入上下文
                        context.add_tool_message(
                            tool_call.tool_call_id,
                            tool_call.name,
                            tool_result
                        )
                    
                    # 继续循环（让 LLM 基于工具结果生成回复）
                    continue
                else:
                    # 没有工具调用，直接返回结果
                    final_response = parsed.final_answer or "抱歉，我没有理解您的请求。"
                    
                    # 记录助手回复
                    context.add_assistant_message(
                        content=final_response,
                        thinking=parsed.thinking
                    )
                    
                    break
            
            else:
                # 达到最大迭代次数
                final_response = "抱歉，我尝试了多次但未能完成任务。请尝试简化您的问题。"
                context.add_assistant_message(content=final_response)
                logger.warning(f"达到最大迭代次数: {self.max_iterations}")
            
            # 5. 记录追踪
            trace_logger.log_agent_response(trace, final_response)
            
            return {
                "response": final_response,
                "session_id": session.session_id,
                "trace_id": trace.trace_id,
                "tool_calls_used": tool_calls_used,
                "turns": context.get_turn_count(),
                "thinking": parsed.thinking if parsed else None
            }
            
        except Exception as e:
            logger.error(f"Agent 执行异常: {str(e)}", exc_info=True)
            error_msg = f"抱歉，处理您的请求时出现了错误: {str(e)}"
            trace.add_entry("error", {"error": str(e)})
            
            return {
                "response": error_msg,
                "session_id": session.session_id,
                "trace_id": trace.trace_id,
                "tool_calls_used": tool_calls_used,
                "turns": context.get_turn_count(),
                "thinking": None
            }
    
    def _execute_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """执行单个工具调用"""
        try:
            result = tool_registry.execute(tool_call.name, **tool_call.arguments)
            return result
        except ValueError as e:
            return {"success": False, "error": f"工具参数错误: {str(e)}"}
        except Exception as e:
            logger.error(f"工具执行异常: {tool_call.name}: {str(e)}")
            return {"success": False, "error": f"工具执行失败: {str(e)}"}
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        session = self.session_manager.get_session(session_id)
        if not session:
            return None
        
        context = self._get_context(session_id)
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at,
            "last_active": session.last_active,
            "turns": context.get_turn_count(),
            "messages_count": len(context.messages)
        }
    
    def clear_session_context(self, session_id: str):
        """清除会话的上下文"""
        if session_id in self._contexts:
            self._contexts[session_id].clear()
            del self._contexts[session_id]
