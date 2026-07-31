"""
مسارات الذكاء الاصطناعي التنبؤي والتحليل الدلالي المعرفي (v6.1).

هذه الطبقة مُدمجة من التطوير الذي أجراه الباحث فوق البنية الأصلية، وأُعيد
تكييفها هنا لتعمل على البنية الديناميكية الفعلية للمنصة (Scale/UserAssessment)
بدل جدول تشخيص ثابت الأعمدة (DiagnosticAssessment) كان موجوداً في نسخة تجريبية
منفصلة. الفكرة: بدل تخزين قيم "الترجمة القيمية" و"أنسنة التكنولوجيا"... إلخ في
أعمدة SQL جامدة، تُقرأ هذه القيم من `UserAssessment.calculated_scores` (JSONB)
حسب أسماء الأبعاد التي عرّفها الباحث عند إنشاء المقياس عبر `/api/scales`
(انظر Scales CMS الديناميكي) - بذلك يبقى مبدأ "مقياس جديد دون تعديل الكود"
سارياً حتى على الطبقة التنبؤية.

جديد في v6.1:
- النموذج يُحمَّل من ملف محفوظ على القرص (`app.ai_engine.predictive_models`)
  بدل إعادة بنائه من الصفر في كل عملية uvicorn.
- مسار `/api/ai/retrain` (admin فقط) يقبل بيانات مُصنَّفة حقيقية ويحفظ
  النموذج الناتج، بدل أن تبقى `retrain()` كوداً ميتاً بلا أي راوت يستدعيها.
- تمييز بين خطأ إعداد حقيقي (بُعد مطلوب غير معرَّف أصلاً في هذا المقياس - يجب
  أن يفشل الطلب بوضوح) وبين نقص طبيعي في نتيجة مستجيب واحد بعينه (يُعامَل
  بقيمة محايدة كما في السابق، لكن دون خلط الحالتين).
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.ai_engine.nlp_sentiment import deep_semantic_cognitive_load_analysis
from app.ai_engine.predictive_models import AdvancedPredictiveEngine
from app.auth import get_current_user, require_role
from app.config import get_settings
from app.database import get_db
from app.models import Scale, User, UserAssessment
from app.schemas import RetrainRequest, SemanticAnalysisRequest

router = APIRouter(prefix="/api/ai", tags=["AI Insights (v6.1)"])

settings = get_settings()

# نموذج واحد مشترك لعمر العملية (uvicorn worker) كله، لكن مُحمَّل من ملف مشترك
# على القرص بدل بيانات bootstrap طازجة في كل عملية - راجع docker-compose.prod.yml
# لمسار الـ volume المشترك (ai_model_store) المتوافق مع read_only: true.
_predictive_engine = AdvancedPredictiveEngine.load_or_create(settings.ai_model_store_path)

# أسماء الأبعاد الافتراضية التي يتوقعها إطار "الترجمة القيمية" (Value
# Translation) لدى الباحث. يمكن للمستخدم تمرير أسماء أبعاد مختلفة عبر معامل
# `dimensions` إن كان قد سمّى أبعاد مقياسه بشكل مغاير.
DEFAULT_RISK_DIMENSIONS = [
    "الترجمة القيمية",
    "أنسنة التكنولوجيا",
    "التوافق الإنساني الذكي",
    "السلامة التشغيلية",
]

RETRAIN_FEATURE_ORDER = (
    "value_translation_score",
    "humanizing_tech_score",
    "human_smart_compatibility",
    "operational_safety_score",
)


@router.post("/semantic-analysis")
def semantic_analysis(
    payload: SemanticAnalysisRequest,
    user: User = Depends(get_current_user),
):
    """تحليل دلالي/حمل معرفي لدفعة من الملاحظات النصية الحرة (مثلاً ملاحظات
    الموظفين المفتوحة المرفقة بتقييم ما)."""
    return deep_semantic_cognitive_load_analysis(payload.feedback_texts)


@router.get("/predict-risk/{scale_id}")
def predict_risk(
    scale_id: int,
    dimensions: Optional[List[str]] = Query(
        default=None,
        description=(
            "أسماء الأبعاد الأربعة (بترتيب: الترجمة القيمية، أنسنة التكنولوجيا، "
            "التوافق الإنساني-الذكي، السلامة التشغيلية) كما عُرّفت فعلياً عند "
            "إنشاء هذا المقياس. اختياري؛ تُستخدم الأسماء الافتراضية إن لم تُمرَّر."
        ),
    ),
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin", "researcher")),
):
    """
    يستخدم درجات الأبعاد الفعلية المخزَّنة (`UserAssessment.calculated_scores`)
    لهذا المقياس لتقدير احتمال الخطر التنظيمي عبر محرك التعلم الآلي (Random
    Forest).

    التعامل مع القيم الناقصة (v6.1، أدق من السابق):
    - إن كان أحد الأبعاد الأربعة المطلوبة غير مُعرَّف أصلاً ضمن أبعاد هذا
      المقياس (خطأ إعداد فعلي من الباحث عند إنشاء المقياس أو تمرير أسماء
      أبعاد خاطئة) → يفشل الطلب بوضوح (422) بدل تمرير الخطأ بصمت.
    - إن كان البُعد مُعرَّفاً في المقياس لكن مستجيباً بعينه لم يُجب عليه (حالة
      طبيعية) → يُستخدم القيمة المحايدة 3.0 لتلك الحالة فقط، ويُبلَّغ عددها.

    مقصور على admin/researcher (وليس أي مستخدم مصادَق عليه): هذه النقطة تُجمِّع
    درجات كل المستجيبين على مستوى المقياس بأكمله، وهو ما يماثل حساسية إنشاء/إدارة
    المقاييس نفسها في `app/routers/scales.py` - إتاحته لأي respondent كانت تسمح
    له بالاطلاع على مؤشرات تنظيمية مجمَّعة تخص مستجيبين آخرين لا علاقة له بهم.
    """
    scale = db.get(Scale, scale_id)
    if not scale:
        raise HTTPException(status_code=404, detail="المقياس غير موجود")

    dims = dimensions or DEFAULT_RISK_DIMENSIONS
    if len(dims) != 4:
        raise HTTPException(status_code=422, detail="يجب تحديد أربعة أبعاد بالضبط للتنبؤ.")

    # خطأ إعداد حقيقي: بُعد مطلوب غير معرَّف أصلاً في هذا المقياس. نفشل بوضوح
    # بدل معاملته كنقص إجابة طبيعي (وإلا فإن كل استجابة ستكون 3.0 بصمت لبُعد
    # غير موجود من الأساس، ما يخفي خطأ إعداد حقيقي عن الباحث).
    configured_dim_names = {d.name for d in scale.dimensions}
    unconfigured = [d for d in dims if d not in configured_dim_names]
    if unconfigured:
        raise HTTPException(
            status_code=422,
            detail=(
                "الأبعاد التالية غير معرَّفة أصلاً في هذا المقياس: "
                f"{', '.join(unconfigured)}. تحقق من أسماء الأبعاد كما أُنشئت "
                "فعلياً في هذا المقياس، أو مرّرها صراحة عبر معامل dimensions."
            ),
        )

    assessments = db.query(UserAssessment).filter(UserAssessment.scale_id == scale_id).all()
    if not assessments:
        raise HTTPException(status_code=404, detail="لا توجد بيانات تقييم كافية لهذا المقياس بعد.")

    respondent_missing_count = 0
    records = []
    for a in assessments:
        record = {}
        record_missing = False
        for feature_key, dim_name in zip(RETRAIN_FEATURE_ORDER, dims):
            value = a.calculated_scores.get(dim_name)
            if value is None:
                record_missing = True
                value = 3.0
            record[feature_key] = value
        if record_missing:
            respondent_missing_count += 1
        records.append(record)

    result = _predictive_engine.evaluate_organizational_risk(records)
    return {
        "status": "success",
        "scale_id": scale_id,
        "sample_size": len(records),
        "respondents_with_missing_dimensions": respondent_missing_count or None,
        "predictive_assessment": result,
    }


@router.post("/retrain")
def retrain_model(
    payload: RetrainRequest,
    user: User = Depends(require_role("admin")),
):
    """
    يُعيد تدريب النموذج التنبؤي على بيانات استبيان حقيقية مُصنَّفة (بدل بيانات
    bootstrap التوضيحية)، ثم يحفظ النموذج الناتج على القرص المشترك حتى تلتقطه
    بقية عمليات uvicorn عند إعادة تشغيلها.

    مقصور على admin فقط: إعادة تدريب النموذج تؤثر على كل استجابات
    `/api/ai/predict-risk` لكل المقاييس والباحثين، وليست عملية بمستوى مقياس
    واحد يمكن تفويضها لـ researcher.

    ملاحظة تشغيلية: هذا يُحدِّث النموذج داخل العملية الحالية فوراً، ويُحفَظ على
    القرص لتلتقطه العمليات الأخرى عند إعادة تشغيلها (لا يوجد بث حي بين عمليات
    uvicorn الحيّة حالياً - إعادة تشغيل تدريجية للخدمة بعد التدريب توصية عملية
    حتى تتّسق كل العمليات على نفس النموذج فوراً).
    """
    import numpy as np

    x = np.array([[getattr(r, k) for k in RETRAIN_FEATURE_ORDER] for r in payload.records])
    y = np.array([r.label for r in payload.records])

    try:
        _predictive_engine.retrain(x, y)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        _predictive_engine.save(settings.ai_model_store_path)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"تم التدريب في الذاكرة لكن تعذّر الحفظ على القرص: {exc}",
        ) from exc

    return {"status": "success", "model_metadata": _predictive_engine.metadata()}
