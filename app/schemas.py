from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------- المستخدم والمصادقة ----------

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None
    role: str = Field(default="respondent")

    @field_validator("role")
    @classmethod
    def restrict_self_registration_role(cls, value: str) -> str:
        # لا يُسمح بمنح صلاحية admin عبر التسجيل الذاتي مهما كانت القيمة المُرسلة
        if value not in ("respondent", "researcher"):
            return "respondent"
        return value


class UserOut(BaseModel):
    id: str
    email: EmailStr
    full_name: str | None
    role: str
    is_active: bool

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    # عبر جسم JSON بدل query parameter - التوكن لا يجب أن يظهر في سجلات
    # الوصول (access logs) الخاصة بـ nginx/uvicorn ولا في تاريخ المتصفح.
    refresh_token: str


# ---------- المقاييس (CMS الديناميكي) ----------

class QuestionCreate(BaseModel):
    question_text: str
    dimension_name: str
    likert_scale_type: str = "standard_5"
    likert_min: int = 1
    likert_max: int = 5
    is_reverse_scored: bool = False


class DimensionCreate(BaseModel):
    name: str
    weight_multiplier: float = 1.0


class ScaleCreate(BaseModel):
    title: str
    description: str | None = None
    dimensions: list[DimensionCreate]
    questions: list[QuestionCreate]

    @field_validator("questions")
    @classmethod
    def questions_reference_known_dimensions(cls, questions, info):
        dims = {d.name for d in info.data.get("dimensions", [])}
        for q in questions:
            if q.dimension_name not in dims:
                raise ValueError(
                    f"السؤال يشير إلى بُعد غير معرّف: {q.dimension_name}"
                )
        return questions


class QuestionOut(BaseModel):
    id: int
    question_text: str
    likert_scale_type: str
    likert_min: int
    likert_max: int

    class Config:
        from_attributes = True


class DimensionOut(BaseModel):
    id: int
    name: str
    weight_multiplier: float
    questions: list[QuestionOut] = []

    class Config:
        from_attributes = True


class ScaleOut(BaseModel):
    id: int
    title: str
    description: str | None
    creator_type: str
    created_by_id: str | None
    is_active: bool
    dimensions: list[DimensionOut] = []

    class Config:
        from_attributes = True


# ---------- التشخيص/التقييم ----------

class AssessmentSubmit(BaseModel):
    scale_id: int
    # answers: {question_id (كنص): قيمة ليكرت}
    answers: dict[str, int]
    request_ai_interpretation: bool = False


class AssessmentOut(BaseModel):
    id: int
    scale_id: int
    calculated_scores: dict
    ai_interpretation: str | None
    current_stage: str
    created_at: datetime

    class Config:
        from_attributes = True


# ---------- الذكاء الاصطناعي التنبؤي والتحليل الدلالي (v6.0) ----------

class SemanticAnalysisRequest(BaseModel):
    feedback_texts: list[str] = Field(min_length=1, max_length=500)

    @field_validator("feedback_texts")
    @classmethod
    def texts_not_all_blank(cls, texts: list[str]) -> list[str]:
        if not any(t.strip() for t in texts):
            raise ValueError("يجب أن يحتوي طلب واحد على الأقل على نص فعلي غير فارغ")
        return texts


class RiskPredictionOut(BaseModel):
    status: str
    scale_id: int
    sample_size: int
    predictive_assessment: dict


class RetrainRecord(BaseModel):
    """سجل تدريب واحد: القيم الأربعة الفعلية لأبعاد "الترجمة القيمية" مع
    الوسم الحقيقي (0 = لا مخاطر، 1 = خطر تنظيمي) الذي حدّده باحث/خبير مُختص -
    وليس نتاج النموذج نفسه (تجنّباً لحلقة تغذية راجعة ذاتية التحيّز)."""

    value_translation_score: float = Field(ge=0, le=10)
    humanizing_tech_score: float = Field(ge=0, le=10)
    human_smart_compatibility: float = Field(ge=0, le=10)
    operational_safety_score: float = Field(ge=0, le=10)
    label: int = Field(description="0 = لا مخاطر، 1 = خطر تنظيمي")

    @field_validator("label")
    @classmethod
    def label_is_binary(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("label يجب أن تكون 0 أو 1 فقط")
        return value


class RetrainRequest(BaseModel):
    records: list[RetrainRecord] = Field(
        min_length=10,
        description="10 سجلات مُصنَّفة حقيقية على الأقل - أقل من ذلك يُعيد إنتاج "
        "مشكلة نموذج bootstrap الذي يحفظ العيّنات بدل التعلّم منها.",
    )

    @field_validator("records")
    @classmethod
    def both_classes_present(cls, records: list["RetrainRecord"]) -> list["RetrainRecord"]:
        labels = {r.label for r in records}
        if len(labels) < 2:
            raise ValueError("يجب أن تحتوي بيانات التدريب على عيّنات من الفئتين (0 و1) معاً")
        return records


class RetrainResponseOut(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_metadata: dict
