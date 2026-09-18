# DeepResearch · 多智能体深度研究系统

基于 LangGraph 的多智能体研究助手。输入一个开放式问题，自动走完 **意图识别 → 规划 → 双路检索 → 深度分析 → 反思迭代 → 报告生成** 全流程，输出带可追溯来源的研究报告。

技术栈：Python · LangGraph · LangChain · DashScope(Qwen) · Milvus · Bocha · FastAPI · Vue 3

---

## 一、架构

**Pipeline（串行阶段）拓扑**，8 个 Agent 串联，中间嵌一个带硬上限的反思循环。

```mermaid
flowchart TB
    START([用户提问]) --> IR{"intent_router<br/>关键词规则 + LLM 二次确认"}
    IR -->|"简单问题"| DA["direct_answer"] --> END([直接回答])
    IR -->|"调研类问题"| PL["planner<br/>拆解子问题"]
    PL --> WS["web_search<br/>Bocha 联网检索"]
    PL --> LR["local_rag<br/>Milvus 向量检索"]
    WS --> DD["deep_dive"] --> AN["analyze<br/>证据裁决"]
    LR --> DD
    AN -->|"证据不足<br/>且未达迭代上限"| RF["reflect<br/>生成补充检索计划"]
    RF --> WS
    RF --> LR
    AN -->|"证据充分"| WR["writer<br/>带引用校验"] --> END
```

- **8 个 Agent**：`intent_router`、`planner`、`scout_web`、`scout_local`、`evidence_judge`、`analyst`、`direct_responder`、`writer`
- **反思循环硬上限**：由 `MAX_ITERATIONS` 控制（默认 3），防止无限重搜
- 图定义见 `app/mult_agents/graph.py`，节点逻辑见 `app/mult_agents/nodes.py`

---

## 二、设计要点

### 1. 零工具模式（Agent 不绑定工具）

所有 Agent 通过 `create_agent(tools=[])` 创建，**不注册任何 LangChain 工具**。实际的检索动作由节点函数直接调用 `tools.py` 里的普通 Python 函数完成。

这样做的收益：LLM 只负责它擅长的部分（判断、归纳、生成），而参数拼装、异常处理、重试、超时全部留在确定性的 Python 里。Agent 在这里的角色是**结构化输出解析器**，不是 tool-calling agent。

### 2. 证据可追溯，防止编造引用

每次检索都记录 `web_search_trace` / `local_rag_trace`，包含 `raw_source_ids`、`kept_source_ids`、`rejected_source_ids`。`writer` 节点会校验生成的引用是否落在 `valid_source_ids` 集合内 —— 模型引用了不存在的来源会被拦下。

### 3. 每个 LLM 节点都有降级路径

所有调用 LLM 的节点都配有对应的 Python 降级函数（`_fallback_analysis`、`_fallback_audit` 等）。即使模型返回格式错误的 JSON，Pipeline 仍能产出完整报告，而不是抛异常中断。

### 4. 40+ 字段的共享状态

`ResearchState` 在阶段之间显式传递所有中间产物（`plan`、`web_evidence`、`audit`、`evidence_pool`、`analysis`、`draft`…）。

这是 Pipeline 模式的固有代价：阶段之间没有隐式上下文，**每一处交接都必须建模成状态字段**。对比 Router 模式只需要的少数几个字段，这个差异来自拓扑本身，不是设计冗余。

### 5. 消息不跨节点累积

`_invoke_json_agent()` 每次只把当前节点的 `HumanMessage` 传给模型（外加 `with_memory_context()` 注入的记忆段落），不累加上游节点的对话历史。目的是控制 Token 消耗，并让模型专注于当前节点单一任务。

### 6. 双层记忆 + 多后端降级

- **短期**：对话缓冲，超过 `SHORT_TERM_MAX_MESSAGES`（默认 30）后由 LLM 递归压缩为摘要
- **长期**：语义记忆（事实、用户画像）+ 情景记忆（任务历史）
- **降级链**：Postgres → Redis → SQLite/内存，逐级回退
- **Checkpointer**：支持长任务中断后续跑

---

## 三、快速开始

### 环境要求

Python 3.10+、Node.js 18+。需要 DashScope API Key（必填）；Milvus 与 Bocha 可选。

### 后端（FastAPI · 端口 8000）

```bash
cd deep_research

cp .env.example .env
# 编辑 .env，至少填写 DASHSCOPE_API_KEY
# 如需联网检索，再填 BOCHA_API_KEY

pip install -r requirements.txt

python main.py            # CLI 交互模式
python main.py --once-query "你的问题"   # 单次执行
python app/app_main.py    # API 服务模式，监听 0.0.0.0:8000
```

### 前端（Vite · 端口 5173）

```bash
cd front/agent_front
npm install
npm run dev
```

前端 dev server 已配置代理，`/api` 请求会转发到 `http://127.0.0.1:8000`。

### 知识库入库（可选，需要 Milvus）

`local_rag` 节点从 Milvus 的 `MILVUS_COLLECTION` 集合检索。把本地文档灌进去：

```bash
python app/mult_agents/rag/ingest.py <文件或目录>   # 目录会递归收集 *.txt/*.md/*.markdown
```

不灌数据时 `local_rag` 检索为空，Pipeline 会靠 Bocha 联网检索这一路继续跑通。

### 配置优先级

**环境变量 > `config.json` > 代码默认值**（实现见 `app/mult_agents/config.py` 的 `_resolve_str`）。

`config.json` 可覆盖：模型名、最大迭代次数、记忆后端、Milvus 开关、Checkpointer 后端。

---

## 四、目录结构

```
app/
├── mult_agents/          # LangGraph 核心
│   ├── graph.py          # StateGraph 构建：节点、条件边、反思循环
│   ├── nodes.py          # 9 个节点函数
│   ├── state.py          # ResearchState（40+ 字段）
│   ├── prompts.py        # 17 个系统提示词，其中 8 个接线到实际 Agent
│   ├── tools.py          # Bocha 检索、Milvus RAG 封装
│   ├── config.py         # AppConfig 不可变 dataclass
│   ├── memory/           # 双层记忆 + 多后端降级链
│   └── rag/              # 向量检索核心 + ingest.py 入库 CLI
├── backend/              # FastAPI 层
│   ├── router/           # /health、/api/v1/research
│   ├── service/          # WorkflowService，支持同步与 SSE 流式
│   └── schemas/          # 请求/响应模型
└── app_main.py           # API 服务入口

front/agent_front/        # Vue 3 前端（SSE 流式渲染）
```

---

## 五、已知限制

- **无自动化测试**。项目定位是架构演示，代码质量重点在模式而非生产级健壮性。
- **多数工具是桩函数**。`tools.py` 中除 Bocha 联网检索与 Milvus RAG 外的工具（文件系统、SQL、地图等）为占位实现。
- **完整记忆功能依赖外部服务**。Postgres / Redis / Milvus 未接入时会逐级降级到 SQLite，功能可用但非预期路径。
- **反思循环的收敛收益未经量化**。当前只能观察到"通常 1–2 轮即满足证据充分条件"，缺少对照实验数据。
- **`prompts.py` 有 9 个提示词未接线**。17 个 key 中只有 8 个被 `build_agent()` 实际加载；`reflect`、`codegen`、`sql_agent` 等属于遗留内容。其中 `reflect` 值得注意——reflect 节点在 `graph.py` 里复用了 `agents.planner`，因此它执行时用的是 planner 的 system prompt，而非那份专门为补搜计划写的 reflect prompt。
- **SQLite 降级后端无并发保护**，仅适用于本地单进程使用。
