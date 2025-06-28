"""Configuration module for loading environment variables."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Database Configuration
    DB_NAME = os.getenv("DB_NAME", "gmail_article_search")
    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASS = os.getenv("DB_PASS", "postgres")
    DB_HOST = os.getenv("DB_HOST", "db")
    DB_PORT = os.getenv("DB_PORT", "5432")
    
    # Gmail API Configuration
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    # MCP Configuration
    MCP_SERVER_TIMEOUT = int(os.getenv("MCP_SERVER_TIMEOUT", "30"))
    MCP_SERVER_RETRIES = int(os.getenv("MCP_SERVER_RETRIES", "3"))
    
    # Service Configuration
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
    
    # Embedding Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Application Configuration
    VECTOR_TABLE_NAME = os.getenv("VECTOR_TABLE_NAME", "medium_articles")
    MEMORY_FILE_PATH = os.getenv("MEMORY_FILE_PATH", "/app/last_update.txt")
    
    # Local testing configuration
    if not os.path.exists(os.path.dirname(MEMORY_FILE_PATH)):
        MEMORY_FILE_PATH = os.path.join(os.getcwd(), "last_update.txt")
    
    # Ollama LLM Configuration
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    
    @property
    def DATABASE_URL(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

# Global config instance
config = Config()
