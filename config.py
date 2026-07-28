"""
Agent 配置文件
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# 加载 .env 文件
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv 未安装时跳过


@dataclass
class AgentConfig:
    """Agent 配置"""
    # LLM 配置
    llm_api_key: str = field(default_factory=lambda: (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ))
    llm_base_url: str = field(default_factory=lambda: (
        os.getenv("OPENAI_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))
    llm_model: str = field(default_factory=lambda: (
        os.getenv("LLM_MODEL_NAME")
        or os.getenv("LLM_MODEL")
        or "qwen-max"
    ))
    
    # Context 配置
    max_turns: int = 10  # 最大对话轮次
    max_context_tokens: int = 4000  # 最大上下文 token 数
    compression_threshold: int = 3000  # 触发压缩的阈值
    
    # Session 配置
    session_timeout: int = 3600  # session 超时时间（秒）
    
    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = "agent.log"


# 全局配置实例
config = AgentConfig()
