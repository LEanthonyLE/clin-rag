# Agentic LightRAG

这是一个基于 LightRAG 的 Agentic RAG (Retrieval-Augmented Generation) 系统，旨在提供更智能的信息检索和生成能力。

## 目录结构

```
clin-rag-main/
├── src/agentic_lightrag/       # 源码包
│   ├── api/                    # FastAPI 接口 (main.py)
│   ├── orchestrator/           # 流水线编排 (simple_pipeline.py)
│   ├── skills/                 # 技能实现 (base / implementations / stubs)
│   ├── schemas/                # Pydantic 数据模型 (common / contracts)
│   ├── utils/                  # 外部客户端 (llm.py / lightrag_client.py)
│   └── config.py               # 全局配置 (环境变量)
├── pyproject.toml              # 项目元数据与依赖 (uv 管理)
├── uv.lock                     # 依赖锁定文件
├── main.py                     # 项目入口占位
├── README.md
└── .gitignore
```

## 环境配置

本项目使用环境变量进行配置。请复制 `.env` 文件（如果不存在请创建）并设置以下变量：

> **注意**: 请勿将包含真实 API Key 的 `.env` 文件提交到版本控制系统中。

```ini
# LightRAG API 设置
LIGHTRAG_API_BASE_URL="http://localhost:9621"
LIGHTRAG_API_KEY="your_lightrag_key"

# 数据库设置
MILVUS_URI="http://localhost:19530"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_password"

# LLM 设置
LLM_API_KEY="your_llm_api_key"
LLM_BASE_URL="your_llm_base_url"
LLM_MODEL="gpt-4o"

# 其他设置
WORKSPACE="default"
```

## 运行

```bash
# 安装依赖 (自动创建 .venv 并安装项目)
uv sync

# 启动服务
uv run python -m agentic_lightrag.api.main
```
