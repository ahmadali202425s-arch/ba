"""
وحدة التحليل الدلالي المعرفي (Semantic Cognitive Load Analysis) - v6.0
------------------------------------------------------------------
منقولة ومُدمجة من الطبقة التنبؤية التي طوّرها الباحث فوق البنية الأساسية
للمنصة. تعتمد منهجاً قاموسياً (Lexicon-based) خفيفاً لا يتطلب نماذج خارجية
أو تنزيلات إضافية، بحيث تعمل فوراً دون بنية تحتية إضافية.

يمكن لاحقاً استبدالها بنموذج NLP عربي حقيقي (مثل AraBERT عبر transformers)
دون تغيير توقيع الدالة العامة `deep_semantic_cognitive_load_analysis`، حتى لا
ينكسر أي كود يستدعيها (راجع app/routers/ai_insights.py).
"""

import re
from typing import Any, Dict, List

# كلمات دالة على الحمل المعرفي المرتفع / الإنهاك
_HIGH_COGNITIVE_LOAD_MARKERS = [
    "إرهاق", "ارهاق", "ضغط", "إنهاك", "انهاك", "متعب", "مرهق", "قلق",
    "توتر", "صعب", "معقد", "مستحيل", "لا أستطيع", "لا استطيع",
    "خوف", "فوضى", "ضياع", "احتراق", "استنزاف", "عبء",
]

# كلمات دالة على المشاعر الإيجابية / الاستقرار
_POSITIVE_MARKERS = [
    "ممتاز", "جيد", "رضا", "مرتاح", "سعيد", "واضح", "منظم", "دعم",
    "تحسن", "سهل", "مريح", "فعال", "ثقة", "متوازن", "إيجابي",
]

# كلمات دالة على المشاعر السلبية
_NEGATIVE_MARKERS = [
    "سيء", "سيئ", "غضب", "إحباط", "احباط", "رفض", "مشكلة", "فشل",
    "خلل", "تعقيد", "استياء", "إهمال", "اهمال", "ظلم",
]

# حد أقصى لطول النص المقبول لكل عنصر (حماية بسيطة من إساءة استخدام النقطة
# لإرسال نصوص ضخمة جداً تستهلك موارد الخادم دون داعٍ)
_MAX_TEXT_LENGTH = 5000


def _normalize(text: str) -> str:
    return re.sub(r"[إأآا]", "ا", text)  # تبسيط الألف لتوحيد المطابقة


def _score_text(text: str) -> Dict[str, Any]:
    text = text[:_MAX_TEXT_LENGTH]
    normalized = _normalize(text)

    load_hits = sum(1 for w in _HIGH_COGNITIVE_LOAD_MARKERS if w in normalized)
    pos_hits = sum(1 for w in _POSITIVE_MARKERS if w in normalized)
    neg_hits = sum(1 for w in _NEGATIVE_MARKERS if w in normalized)

    word_count = max(len(normalized.split()), 1)
    cognitive_load_score = min(5.0, round((load_hits / word_count) * 20 + load_hits * 0.5, 2))

    sentiment_balance = pos_hits - neg_hits
    if sentiment_balance > 0:
        sentiment = "إيجابي"
    elif sentiment_balance < 0:
        sentiment = "سلبي"
    else:
        sentiment = "محايد"

    return {
        "text_preview": text[:80] + ("…" if len(text) > 80 else ""),
        "sentiment": sentiment,
        "cognitive_load_score": cognitive_load_score,
        "flagged_markers": {
            "high_load": load_hits,
            "positive": pos_hits,
            "negative": neg_hits,
        },
    }


def deep_semantic_cognitive_load_analysis(feedback_texts: List[str]) -> Dict[str, Any]:
    """يحلل دفعة من النصوص (ملاحظات الموظفين مثلاً) لتقدير الحمل المعرفي
    والمشاعر العامة، ويعيد ملخصاً إجمالياً بالإضافة إلى تفصيل لكل نص."""
    if not feedback_texts:
        return {"status": "error", "message": "لا توجد نصوص للتحليل."}

    # حماية: تجاهل العناصر الفارغة/غير النصية بدل فشل الطلب بأكمله
    cleaned = [t.strip() for t in feedback_texts if isinstance(t, str) and t.strip()]
    if not cleaned:
        return {"status": "error", "message": "جميع النصوص المرسلة فارغة."}

    per_text_results = [_score_text(t) for t in cleaned]

    avg_load = round(sum(r["cognitive_load_score"] for r in per_text_results) / len(per_text_results), 2)
    sentiment_counts = {"إيجابي": 0, "سلبي": 0, "محايد": 0}
    for r in per_text_results:
        sentiment_counts[r["sentiment"]] += 1

    overall_alert = (
        "تنبيه: مستوى حمل معرفي مرتفع عبر عينة الملاحظات، يُنصح بمراجعة أعباء العمل وقنوات الدعم النفسي."
        if avg_load >= 2.5
        else "المستوى العام للحمل المعرفي ضمن الحدود المستقرة."
    )

    return {
        "status": "success",
        "sample_size": len(per_text_results),
        "average_cognitive_load_score": avg_load,
        "sentiment_distribution": sentiment_counts,
        "overall_alert": overall_alert,
        "details": per_text_results,
    }
