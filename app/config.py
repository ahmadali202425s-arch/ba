"""
إعدادات التطبيق المركزية - تُقرأ من متغيرات البيئة فقط (لا قيم مبيتة/hardcoded).
"""
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # عام
    environment: str = "production"
    debug: bool = False
    project_name: str = "COESIS API"
    domain: str = "yourdomain.com"

    # أمان
    secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # قاعدة البيانات
    database_url: str

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # CORS
    allowed_origins: str = "https://yourdomain.com"

    # الذكاء الاصطناعي (المساعد الذكي)
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # الدفع (Stripe)
    stripe_api_key: str | None = None
    stripe_webhook_secret: str | None = None

    # المراقبة
    sentry_dsn: str | None = None

    # طبقة الذكاء الاصطناعي التنبؤي (v6.0) - مسار استمرار النموذج على القرص
    # (يجب أن يشير إلى مجلد مشترك بين عمليات uvicorn المتعددة، انظر docker-compose.prod.yml)
    ai_model_store_path: str = "/app/model_store/predictive_model.joblib"

    @field_validator("database_url")
    @classmethod
    def _normalize_database_url(cls, v: str) -> str:
        """
        Render يوفر رابط قاعدة البيانات بصيغة postgres:// أو postgresql://
        بينما SQLAlchemy + psycopg2 يتطلبان بادئة صريحة postgresql+psycopg2://
        هذا التحقق يطبّع الصيغة تلقائياً بدل الاعتماد على أن يفعلها المستخدم يدوياً.
        """
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+psycopg2://", 1)
        if v.startswith("postgresql://") and "+psycopg2" not in v:
            return v.replace("postgresql://", "postgresql+psycopg2://", 1)
        return v

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
