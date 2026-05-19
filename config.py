import os

# Load settings (simple implementation without extra dependencies like pydantic-settings for now)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pydantic import BaseModel

class Settings(BaseModel):
    # LightRAG API
    lightrag_api_base_url: str = os.getenv("LIGHTRAG_API_BASE_URL", "http://localhost:9621")
    lightrag_api_key: str = os.getenv("LIGHTRAG_API_KEY", "")

    # Databases
    milvus_uri: str = os.getenv("MILVUS_URI", "http://localhost:19530")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_auth: tuple = (
        os.getenv("NEO4J_USERNAME", "neo4j"),
        os.getenv("NEO4J_PASSWORD", "your_password")
    )

    # App
    default_workspace: str = os.getenv("WORKSPACE", "default")

    # LLM
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o")


settings = Settings()
