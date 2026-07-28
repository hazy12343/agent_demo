"""
pytest 配置文件
"""
import os
import sys

# 设置环境变量避免 OpenAI 初始化错误
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-testing")

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
