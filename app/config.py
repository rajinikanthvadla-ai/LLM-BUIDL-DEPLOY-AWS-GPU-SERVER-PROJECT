from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    app_name: str = "domain-llm-api"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_path: str = ""
    max_tokens: int = 256
    temperature: float = 0.7
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True


settings = Settings()
