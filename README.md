# Mini Agent

一个基于大语言模型的轻量级 Agent 系统，支持工具调用、多轮对话和会话管理。

## 功能特性

- **Agent 循环**：接收输入 → LLM 判断是否调用工具 → 执行工具 → 返回结果
- **工具系统**：内置计算器、搜索、天气查询工具，支持自定义扩展
- **多轮对话**：基于 Session 的上下文管理，支持会话超时自动清理
- **执行追踪**：完整的调用链追踪，记录 LLM 调用和工具执行耗时
- **REST API**：基于 FastAPI 提供 HTTP 接口

## 项目结构

```
agent_demo/
├── agent.py          # Agent 核心运行时
├── config.py         # 配置管理
├── context.py        # 上下文管理器
├── llm_client.py     # LLM 客户端封装
├── logger.py         # 日志与执行追踪
├── parser.py         # LLM 输出解析器
├── server.py         # FastAPI 服务入口
├── session.py        # 会话管理
├── tools/            # 工具模块
│   ├── base.py       # 工具基类与注册表
│   ├── calculator.py # 计算器工具
│   ├── search.py     # 搜索工具
│   └── weather.py    # 天气查询工具
└── tests/            # 测试用例
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=your-api-key-here
LLM_MODEL_NAME=qwen-max
```

支持兼容 OpenAI 接口的任意 LLM 服务，通过 `OPENAI_BASE_URL` 配置。

### 3. 启动服务

```bash
python server.py
```

服务默认运行在 `http://localhost:8000`。

## API 接口

### 对话

```bash
POST /chat
{
    "message": "北京今天天气怎么样？",
    "user_id": "user1"
}
```

### 会话管理

| 方法   | 路径                      | 说明             |
| ------ | ------------------------- | ---------------- |
| GET    | `/sessions/{user_id}`     | 列出用户所有会话 |
| GET    | `/session/{session_id}`   | 获取会话详情     |
| DELETE | `/session/{session_id}`   | 删除会话         |

### 执行追踪

| 方法 | 路径                 | 说明             |
| ---- | -------------------- | ---------------- |
| GET  | `/traces`            | 获取追踪记录列表 |
| GET  | `/traces/{trace_id}` | 获取单条追踪记录 |

### 健康检查

```bash
GET /health
```

## 运行测试

```bash
pytest
```

## 技术栈

- **LLM**：OpenAI SDK（兼容 DashScope / 其他 OpenAI 兼容接口）
- **Web 框架**：FastAPI + Uvicorn
- **数据校验**：Pydantic v2
- **测试**：pytest + httpx
