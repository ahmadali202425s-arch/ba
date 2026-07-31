"""
خدمة المساعد الذكي: تفسير نتائج المقياس عبر OpenAI.
مصممة بحيث لا تُسقط الخادم إن غاب المفتاح أو تعطلت الخدمة الخارجية - فشل هذه
الخدمة يجب ألا يمنع تخزين نتائج التقييم أساساً (لذلك تُستدعى دائماً في الخلفية).
"""
import logging

from openai import OpenAI, OpenAIError

from app.config import get_settings

logger = logging.getLogger("coesis.ai_assistant")
settings = get_settings()

_client: OpenAI | None = None


def _get_client() -> OpenAI | None:
    global _client
    if not settings.openai_api_key:
        return None
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def generate_interpretation(scores: dict) -> str | None:
    client = _get_client()
    if client is None:
        logger.warning("OPENAI_API_KEY غير مضبوط - تم تخطي التفسير الذكي")
        return None

    prompt = (
        "أنت مساعد نفسي متخصص. فسّر نتائج المقياس النفسي التالية للمستخدم بإيجاز "
        "ومهنية دون تشخيص طبي، واقترح خطوة تالية عملية واحدة فقط.\n"
        f"النتائج حسب الأبعاد: {scores}"
    )
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            timeout=20,
        )
        return response.choices[0].message.content
    except OpenAIError as exc:
        logger.error("فشل استدعاء OpenAI: %s", exc)
        return None
