"""
工具基础类和注册机制
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self):
        self.name: str = ""
        self.description: str = ""
        self.parameters: List[ToolParameter] = []
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """执行工具，返回结果字典"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """获取工具 Schema（OpenAI function calling 格式）"""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def validate_params(self, **kwargs) -> bool:
        """验证参数"""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                raise ValueError(f"缺少必需参数: {param.name}")
        return True


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """注册工具"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """获取所有工具的 Schema"""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def execute(self, name: str, **kwargs) -> Dict[str, Any]:
        """执行工具"""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"工具不存在: {name}")
        
        tool.validate_params(**kwargs)
        return tool.execute(**kwargs)


# 全局工具注册表
tool_registry = ToolRegistry()
