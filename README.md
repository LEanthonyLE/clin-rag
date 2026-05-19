# Agentic LightRAG

这是一个基于 LightRAG 的 Agentic RAG (Retrieval-Augmented Generation) 系统，旨在提供更智能的信息检索和生成能力。

## 目录结构

- `api/`: API 接口定义
- `orchestrator/`: 编排层，包含管道定义
- `skills/`: 各种技能实现
- `schemas/`: 数据模型定义
- `utils/`: 工具函数

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

确保已安装必要的依赖项，然后运行：

```bash
python -m api.main
```
(根据实际入口文件调整)
