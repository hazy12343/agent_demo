"""
LLM 客户端
"""
import json
import logging
from typing import Any, Dict, List, Optional
from openai import OpenAI
from config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM 客户端 - 封装 OpenAI API 调用"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self._api_key = api_key or config.llm_api_key
        self._base_url = base_url or config.llm_base_url
        self.model = model or config.llm_model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url
            )
        return self._client
    
    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            tools: 可用工具列表（function calling 格式）
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            
        Returns:
            API 响应
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            logger.debug(f"LLM 请求: model={self.model}, messages_count={len(messages)}")
            
            response = self.client.chat.completions.create(**kwargs)
            
            choice = response.choices[0]
            message = choice.message
            
            result = {
                "content": message.content,
                "tool_calls": None,
                "finish_reason": choice.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
            # 解析工具调用
            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                    }
                    for tc in message.tool_calls
                ]
            
            logger.debug(f"LLM 响应: finish_reason={result['finish_reason']}, "
                        f"tool_calls={result['tool_calls'] is not None}")
            
            return result
            
        except Exception as e:
            logger.error(f"LLM 调用失败: {str(e)}")
            raise


# 全局 LLM 客户端实例（延迟初始化）
llm_client = None

def get_llm_client() -> LLMClient:
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client
