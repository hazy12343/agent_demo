"""
搜索工具（Mock 实现）
"""
from typing import Any, Dict, List
from .base import BaseTool, ToolParameter


class SearchTool(BaseTool):
    """搜索工具 - Mock 实现"""
    
    def __init__(self):
        super().__init__()
        self.name = "search"
        self.description = "搜索信息，返回相关结果"
        self.parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询关键词"
            )
        ]
        
        # Mock 数据库
        self._mock_data = {
            "python": [
                {"title": "Python 官方文档", "url": "https://docs.python.org", "snippet": "Python 是一种高级编程语言..."},
                {"title": "Python 教程", "url": "https://python.org/tutorial", "snippet": "Python 入门教程..."}
            ],
            "agent": [
                {"title": "AI Agent 架构", "url": "https://example.com/agent", "snippet": "Agent 是一种能够感知环境并采取行动的系统..."},
                {"title": "LangChain Agent", "url": "https://langchain.com/agent", "snippet": "LangChain 提供了构建 Agent 的框架..."}
            ],
            "llm": [
                {"title": "大语言模型概述", "url": "https://example.com/llm", "snippet": "LLM 是基于 Transformer 架构的语言模型..."},
                {"title": "GPT 系列模型", "url": "https://openai.com/gpt", "snippet": "GPT 是 OpenAI 开发的生成式预训练模型..."}
            ]
        }
    
    def execute(self, query: str) -> Dict[str, Any]:
        """执行搜索（Mock）"""
        query_lower = query.lower()
        
        # 简单的关键词匹配
        results = []
        for keyword, data in self._mock_data.items():
            if keyword in query_lower:
                results.extend(data)
        
        # 如果没有匹配，返回通用结果
        if not results:
            results = [
                {
                    "title": f"搜索结果: {query}",
                    "url": f"https://example.com/search?q={query}",
                    "snippet": f"这是关于 '{query}' 的模拟搜索结果..."
                }
            ]
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results)
        }
