"""
LLM 输出解析模块
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCall:
    """工具调用"""
    tool_call_id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ParsedOutput:
    """解析后的 LLM 输出"""
    thinking: Optional[str] = None  # 思考过程
    tool_calls: List[ToolCall] = field(default_factory=list)  # 工具调用列表
    final_answer: Optional[str] = None  # 最终答案
    raw_response: Dict[str, Any] = field(default_factory=dict)  # 原始响应


class OutputParser:
    """LLM 输出解析器"""
    
    def parse(self, llm_response: Dict[str, Any]) -> ParsedOutput:
        """
        解析 LLM 响应
        
        支持两种模式：
        1. OpenAI function calling（标准模式）
        2. 文本解析（回退模式）
        """
        result = ParsedOutput(raw_response=llm_response)
        
        content = llm_response.get("content", "")
        tool_calls = llm_response.get("tool_calls")
        
        # 模式1：标准 function calling
        if tool_calls:
            for tc in tool_calls:
                result.tool_calls.append(ToolCall(
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"]
                ))
            # 如果有 tool_calls 但同时有 content，content 作为思考过程
            if content:
                result.thinking = content
            return result
        
        # 模式2：纯文本回复，尝试解析结构化输出
        if content:
            # 尝试提取思考过程（<thinking> 标签）
            thinking_match = re.search(
                r'<thinking>(.*?)</thinking>',
                content,
                re.DOTALL
            )
            if thinking_match:
                result.thinking = thinking_match.group(1).strip()
                content = re.sub(
                    r'<thinking>.*?</thinking>',
                    '',
                    content,
                    flags=re.DOTALL
                ).strip()
            
            # 尝试提取工具调用（JSON 格式）
            tool_call_match = re.search(
                r'```json\s*\n?({\s*"tool":\s*.*?})\s*\n?```',
                content,
                re.DOTALL
            )
            if tool_call_match:
                try:
                    tool_data = json.loads(tool_call_match.group(1))
                    result.tool_calls.append(ToolCall(
                        tool_call_id="text_parsed",
                        name=tool_data.get("tool", ""),
                        arguments=tool_data.get("arguments", {})
                    ))
                    content = re.sub(
                        r'```json\s*\n?{.*?}\s*\n?```',
                        '',
                        content,
                        flags=re.DOTALL
                    ).strip()
                except json.JSONDecodeError:
                    pass
            
            # 剩余内容作为最终答案
            if content:
                result.final_answer = content
        
        return result
    
    def format_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Dict[str, str]:
        """格式化工具执行结果为消息格式"""
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result, ensure_ascii=False, indent=2)
        }
