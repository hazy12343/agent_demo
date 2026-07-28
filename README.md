# Mini Agent — 从零实现的最小可用 Agent 系统

> Vibe Coding 课题：不依赖任何现有 Agent 框架（LangGraph / OpenHands / OpenClaw 等），从零实现完整的 Agent Runtime。

## 目录

- [系统设计](#系统设计)
- [Agent 核心循环](#agent-核心循环)
- [工具系统](#工具系统)
- [Session 管理](#session-管理)
- [Context 管理](#context-管理)
- [异常处理与执行追踪](#异常处理与执行追踪)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 接口](#api-接口)
- [测试用例](#测试用例)
- [技术栈](#技术栈)

---

## 系统设计

### 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                        │
│                    (server.py)                           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  Agent Runtime                           │
│                  (agent.py)                              │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │  LLM     │  │  Output  │  │ Context  │  │ Session│ │
│  │  Client  │  │  Parser  │  │ Manager  │  │ Manager│ │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └───┬────┘ │
│        │             │             │            │       │
└────────┼─────────────┼─────────────┼────────────┼───────┘
         │             │             │            │
         ▼             ▼             ▼            ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ OpenAI   │  │ Thinking │  │ Messages │  │  Multi   │
  │ /DashScope│ │ ToolCall │  │ Compress │  │  Session │
  │ API      │  │ FinalAns │  │ TokenCtl │  │  Store   │
  └──────────┘  └──────────┘  └──────────┘  └──────────┘

         ┌────────────────────────────────┐
         │        Tool Registry           │
         │  ┌─────────┐ ┌──────┐ ┌─────┐ │
         │  │calculator│ │search│ │weath│ │
         │  └─────────┘ └──────┘ └─────┘ │
         └────────────────────────────────┘
```

### 模块职责

| 模块 | 文件 | 职责 |
|------|------|------|
| Agent Runtime | `agent.py` | 核心循环：接收输入 → LLM 决策 → 工具执行 → 循环/返回 |
| LLM Client | `llm_client.py` | 封装 OpenAI 兼容 API，支持 Function Calling |
| Output Parser | `parser.py` | 解析 LLM 输出，提取思考过程、工具调用、最终答案 |
| Context Manager | `context.py` | 消息历史管理、上下文压缩、轮次控制 |
| Session Manager | `session.py` | 多用户多会话隔离、超时清理、线程安全 |
| Tool System | `tools/` | 工具基类、注册表、三个内置工具 |
| Trace Logger | `logger.py` | 执行链路追踪、性能日志 |
| Config | `config.py` | 环境变量与配置管理 |
| Server | `server.py` | FastAPI REST API 入口 |

---

## Agent 核心循环

Agent Runtime 实现了一个 **ReAct 风格**的循环：

```
用户输入
   │
   ▼
┌──────────┐    直接回复    ┌──────────┐
│  LLM 决策 ├──────────────►│ 返回结果  │
│          │               └──────────┘
└─────┬────┘
      │ 需要工具
      ▼
┌──────────┐
│ 执行工具  │
└─────┬────┘
      │ 工具结果
      ▼
┌──────────────┐  继续循环   ┌──────────┐
│ 结果反馈 LLM ├───────────►│  LLM 决策 │
│              │            └──────────┘
└──────┬───────┘
       │ 可以返回
       ▼
┌──────────┐
│ 返回结果  │
└──────────┘
```

**关键设计：**

- 单次请求最大迭代次数 `max_iterations=5`，防止无限循环
- 每次迭代调用 LLM 后，由 Parser 解析输出判断下一步动作
- 工具执行结果自动注入 Context，LLM 可在下一轮迭代中利用

---

## 工具系统

### 内置工具

| 工具 | 名称 | 说明 | 实现方式 |
|------|------|------|----------|
| 计算器 | `calculator` | 支持加减乘除、括号、浮点运算 | 安全沙箱 `eval`，过滤非法字符 |
| 搜索 | `search` | 关键词搜索，返回标题/链接/摘要 | Mock 实现，内置知识库 |
| 天气 | `weather` | 查询城市天气、温度、湿度、建议 | Mock 实现，内置城市数据 |

### 工具注册机制

每个工具继承 `BaseTool`，定义 **名称、描述、参数 Schema**：

```python
class CalculatorTool(BaseTool):
    def __init__(self):
        self.name = "calculator"
        self.description = "执行基本数学运算"
        self.parameters = [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式，例如: '2 + 3 * 4'"
            )
        ]

    def execute(self, expression: str) -> Dict[str, Any]:
        # 安全计算逻辑...
```

工具通过 `ToolRegistry` 全局注册表统一管理：

```python
# 注册
tool_registry.register(CalculatorTool())

# 获取所有 Schema（传递给 LLM 的 tools 参数）
schemas = tool_registry.get_all_schemas()

# LLM 自主决策后执行
result = tool_registry.execute("calculator", expression="2 + 3")
```

生成的 Schema 遵循 **OpenAI Function Calling** 规范，LLM 基于 Schema 自主决策是否调用、调用哪个工具、传什么参数。

### LLM 输出解析

`OutputParser` 支持两种解析模式：

1. **Function Calling 模式（主模式）**：直接解析 `tool_calls` 字段
2. **文本解析模式（回退）**：通过正则提取 `<thinking>` 标签和 JSON 代码块

解析产出三个核心字段：
- `thinking`：LLM 的思考过程
- `tool_calls`：需要执行的工具调用列表
- `final_answer`：最终回复用户的答案

---

## Session 管理

### 独立会话设计

每个 `session_id` 对应独立的 `ContextManager`，实现完全隔离：

```
用户 A ──► 窗口 1 (session_abc) ──► Context_abc ──► 查天气、记待办
       └──► 窗口 2 (session_xyz) ──► Context_xyz ──► 写周报、记待办
```

- 窗口 1 和窗口 2 的对话历史完全独立
- 用户可以随时切回任一窗口继续对话
- Session 通过 `user_id + session_id` 定位
- 不传 `session_id` 时自动创建新会话

### 会话生命周期

- **创建**：首次对话自动创建，生成 UUID 作为 `session_id`
- **续接**：传入已有 `session_id` 可继续对话
- **超时**：默认 3600 秒无活动自动过期
- **清理**：`cleanup_expired()` 批量清理过期会话
- **线程安全**：所有操作通过 `threading.Lock` 保护

---

## Context 管理

### 消息注入策略

Context 按以下顺序构建发送给 LLM 的消息列表：

| 顺序 | 消息类型 | 说明 |
|------|----------|------|
| 1 | System Prompt | 系统指令，定义工具列表和规则 |
| 2 | 压缩摘要 | 历史对话的压缩摘要（如有） |
| 3 | 用户输入 | `role: user` |
| 4 | 助手回复 | `role: assistant`，含 `thinking` 元数据 |
| 5 | 工具调用 | `role: assistant` + `tool_calls` 字段 |
| 6 | 工具结果 | `role: tool` + `tool_call_id` |

**设计考量：**
- 用户输入、工具结果、助手回复均保留在 Context 中，支持**纯对话追问**和**带工具追问**
- 助手的 `thinking`（思考过程）存入 `metadata`，不直接发送给 LLM，避免噪音
- 工具调用消息保留完整的 `tool_calls` 结构，确保 LLM 能理解之前的决策链路

### 压缩策略

当 Context 超过阈值时，执行两级压缩：

1. **轮次压缩**：消息数超过 `max_turns × 2` 时，只保留最近 N 轮
2. **摘要压缩**：token 估算超过 `compression_threshold` 时，将旧消息压缩为 `[历史对话摘要]`，保留最近 3 轮完整对话

```
[历史对话摘要]
用户问过: 北京天气怎么样
助手回答: 北京22°C晴天
使用了工具: weather
[摘要结束]
─── 最近 3 轮完整对话 ───
```

---

## 异常处理与执行追踪

### 异常处理

| 异常场景 | 处理方式 |
|----------|----------|
| LLM API 调用失败 | 捕获异常，返回友好错误消息 |
| 工具不存在 | 返回 `success: False`，继续循环 |
| 工具参数错误 | 返回参数错误提示，LLM 可重试 |
| 达到最大迭代次数 | 返回"尝试多次未完成"提示 |
| Session 过期 | 自动创建新 Session |

### 执行追踪 (Trace)

每次 `chat()` 调用生成一个 `ExecutionTrace`，记录完整执行链路：

```
Trace #a1b2c3d4:
  [user_input]     → "北京天气怎么样？"
  [llm_call]       → messages=3, duration=1200ms
  [tool_call]      → weather(city=北京), duration=2ms, success=True
  [llm_call]       → messages=5, duration=800ms
  [agent_response] → "北京今天22°C..." (total: 2002ms, steps: 5)
```

追踪记录可通过 `/traces` 和 `/traces/{trace_id}` API 查询。

---

## 项目结构

```
agent_demo/
├── agent.py              # Agent 核心运行时（循环逻辑）
├── config.py             # 配置管理（环境变量 + 默认值）
├── context.py            # 上下文管理（消息历史 + 压缩）
├── llm_client.py         # LLM 客户端（OpenAI SDK 封装）
├── logger.py             # 日志与执行追踪
├── parser.py             # LLM 输出解析器
├── server.py             # FastAPI 服务入口
├── session.py            # Session 管理器
├── tools/
│   ├── base.py           # 工具基类 + 注册表
│   ├── calculator.py     # 计算器工具
│   ├── search.py         # 搜索工具（Mock）
│   └── weather.py        # 天气查询工具（Mock）
├── tests/
│   ├── conftest.py       # pytest 配置
│   ├── test_agent.py     # Agent 运行时测试（Mock LLM）
│   ├── test_context.py   # 上下文管理测试
│   ├── test_logger.py    # 追踪日志测试
│   ├── test_parser.py    # 输出解析测试
│   ├── test_server.py    # API 接口测试
│   ├── test_session.py   # Session 管理测试
│   └── test_tools.py     # 工具与注册表测试
├── requirements.txt
└── .env                  # 环境变量（不提交）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```env
# 必填：LLM API Key（支持 DashScope / OpenAI / 兼容接口）
DASHSCOPE_API_KEY=your-api-key-here

# 可选：自定义模型和接口地址
LLM_MODEL_NAME=qwen-max
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 启动服务

```bash
python server.py
```

服务运行在 `http://localhost:8000`，API 文档访问 `http://localhost:8000/docs`。

---

## API 接口

### 对话

```bash
POST /chat
Content-Type: application/json

{
    "message": "北京今天天气怎么样？",
    "user_id": "user_a",
    "session_id": null
}
```

响应示例：

```json
{
    "response": "北京今天22°C，天气晴朗，适合外出。",
    "session_id": "abc123-...",
    "trace_id": "a1b2c3d4",
    "tool_calls_used": [{"tool": "weather", "arguments": {"city": "北京"}, "result": {...}}],
    "turns": 1,
    "thinking": null
}
```

### 会话管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/sessions/{user_id}` | 列出用户所有会话 |
| GET | `/session/{session_id}` | 获取会话详情 |
| DELETE | `/session/{session_id}` | 删除会话 |

### 执行追踪

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/traces` | 获取最近 50 条追踪记录 |
| GET | `/traces/{trace_id}` | 获取单条追踪详情 |

### 健康检查

```bash
GET /health
```

---

## 测试用例

测试使用 **Mock LLM** 模拟各种响应场景，无需真实 API 调用：

```bash
pytest -v
```

### 测试覆盖

| 测试文件 | 覆盖范围 | 用例数 |
|----------|----------|--------|
| `test_agent.py` | 直接回复、三种工具调用、Session 创建/复用/独立、纯对话追问、带工具追问、工具不存在、最大迭代、LLM 异常、Trace 记录 | 15 |
| `test_tools.py` | 计算器八种运算、搜索匹配/兜底、天气多城市/未知城市、注册表 CRUD、Schema 生成、参数校验 | 20 |
| `test_context.py` | 消息增删、轮次压缩、Token 压缩、摘要生成 | 8 |
| `test_session.py` | 创建/获取/删除/超时/清理/线程安全 | 10 |
| `test_parser.py` | Function Calling 解析、文本模式解析、Thinking 提取 | 8 |
| `test_server.py` | 全部 REST API 端点 | 8 |
| `test_logger.py` | Trace 创建/记录/查询 | 6 |

---

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|----------|------|
| LLM 接口 | OpenAI SDK | 兼容 DashScope / 任意 OpenAI 兼容 API |
| Web 框架 | FastAPI + Uvicorn | 异步高性能 REST API |
| 数据校验 | Pydantic v2 | 请求/响应模型 |
| 测试 | pytest + httpx | Mock LLM + API 端到端测试 |
| 配置 | python-dotenv | 环境变量加载 |

---

## Memory（上下文记忆）的召回时机与放置方式

本项目的 "Memory" 通过 **Context Manager** 实现，核心思路是将关键信息按角色注入消息列表，让 LLM 在每次决策时都能获取完整上下文。

### 召回时机

| 时机 | 召回内容 | 放置方式 |
|------|----------|----------|
| **每次 LLM 调用前** | System Prompt | 消息列表第 1 条，`role: system` |
| **每次 LLM 调用前** | 完整对话历史（含用户输入、助手回复、工具调用及结果） | 按时间顺序排列在 System Prompt 之后 |
| **Context 超限时** | 历史摘要 | 压缩为 `[历史对话摘要]`，`role: system`，放在最近 3 轮对话之前 |
| **工具执行后** | 工具返回结果 | `role: tool` + `tool_call_id`，紧跟在对应的 `assistant` 工具调用消息之后 |

### 放置方式设计

```
messages = [
  {role: "system",    content: "你是一个智能助手..."},     ← System Prompt（始终在头部）
  {role: "system",    content: "[历史对话摘要]..."},      ← 压缩摘要（超限后出现）
  {role: "user",      content: "北京天气怎么样"},         ← 用户输入
  {role: "assistant", content: "", tool_calls: [...]},    ← 助手的工具调用决策
  {role: "tool",      content: "{温度: 22°C...}"},       ← 工具执行结果
  {role: "assistant", content: "北京22°C晴天..."},        ← 助手最终回复
  {role: "user",      content: "那上海呢"},               ← 追问（利用上文）
  ...
]
```

**关键设计决策：**
- 助手的 `thinking`（思考过程）仅存入 `metadata`，**不发送给 LLM**，避免干扰后续决策
- 工具调用保留完整的 `tool_calls` 结构（含 id/name/arguments），确保 LLM 能理解之前的决策链路
- 工具结果序列化为 JSON 字符串，保持结构化信息不丢失

---

## AI Prompt 与问题解决记录

### 开发过程中的 AI 辅助

本项目全程使用 AI 辅助开发（Vibe Coding），以下是关键 Prompt 和解决的问题：

| 问题 | AI 辅助方式 | 解决方案 |
|------|-------------|----------|
| Agent 循环如何避免无限死循环？ | 与 AI 讨论 ReAct 架构 | 设置 `max_iterations` 上限，达到后强制返回 |
| 工具调用结果如何让 LLM 理解？ | 询问 OpenAI Function Calling 最佳实践 | 使用标准 `role: tool` + `tool_call_id` 消息格式 |
| Context 过长怎么办？ | 讨论压缩策略 | 实现两级压缩：轮次截断 + 摘要压缩 |
| 多 Session 如何隔离？ | 讨论架构设计 | 每个 `session_id` 对应独立的 `ContextManager` 实例 |

### System Prompt 设计

```
你是一个智能助手 Agent。你可以使用以下工具来帮助用户解决问题：
1. calculator - 计算器：执行数学运算
2. search - 搜索：搜索信息
3. weather - 天气：查询城市天气

规则：
- 如果用户的问题需要查询信息或进行计算，请使用对应的工具
- 如果不需要工具，直接回复用户
- 工具返回结果后，请基于结果给出完整的回答
- 保持回复简洁、有帮助
```

**Prompt 设计要点：**
- 明确列出可用工具，辅助 LLM 决策
- 给出清晰的规则：何时用工具、何时直接回复
- 要求基于工具结果回答，而非简单转发原始数据
