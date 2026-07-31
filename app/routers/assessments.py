from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from app.auth import get_current_user
from app.database import get_db
from app.models import AssessmentStage, Scale, ScaleDimension, User, UserAssessment
from app.schemas import AssessmentOut, AssessmentSubmit
from app.services.ai_assistant import generate_interpretation

router = APIRouter(prefix="/api/assessments", tags=["Assessment & Diagnosis"])


@router.post("/submit", response_model=AssessmentOut, status_code=201)
def submit_and_evaluate_assessment(
    payload: AssessmentSubmit,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    محور التشخيص والتحليل والتقييم الكمي الديناميكي.

    الإصلاحات مقارنة بالنسخة الأصلية:
      - ربط التقييم بالمستخدم المصادَق عليه (user_id) بدل تجاهله.
      - التحقق من أن كل سؤال أُجيب عنه فعلاً وأن القيمة ضمن مدى ليكرت الخاص بالسؤال
        (بدل .get(..., 0) الذي كان يُسكت الأخطاء ويحسب الإجابات الناقصة كصفر بصمت).
      - دعم البنود المعكوسة (is_reverse_scored) في حساب الدرجة.
      - تفعيل تفسير الذكاء الاصطناعي بشكل اختياري وغير حاجز (لا يفشل الطلب كله لو
        تعطلت خدمة OpenAI).
    """
    scale = (
        db.query(Scale)
        .options(selectinload(Scale.dimensions).selectinload(ScaleDimension.questions))
        .filter(Scale.id == payload.scale_id)
        .first()
    )
    if not scale or not scale.is_active:
        raise HTTPException(status_code=404, detail="المقياس غير موجود أو غير مفعل")

    calculated_results: dict[str, float] = {}
    missing_questions: list[int] = []
    out_of_range: list[int] = []

    for dim in scale.dimensions:
        dim_total = 0
        for q in dim.questions:
            raw = payload.answers.get(str(q.id))
            if raw is None:
                missing_questions.append(q.id)
                continue
            if not (q.likert_min <= raw <= q.likert_max):
                out_of_range.append(q.id)
                continue
            value = (q.likert_max + q.likert_min - raw) if q.is_reverse_scored else raw
            dim_total += value
        calculated_results[dim.name] = round(dim_total * dim.weight_multiplier, 2)

    if missing_questions:
        raise HTTPException(
            status_code=422,
            detail={"message": "توجد أسئلة لم تتم الإجابة عنها", "question_ids": missing_questions},
        )
    if out_of_range:
        raise HTTPException(
            status_code=422,
            detail={"message": "قيم إجابات خارج مدى مقياس ليكرت المسموح", "question_ids": out_of_range},
        )

    assessment = UserAssessment(
        user_id=user.id,
        scale_id=scale.id,
        raw_answers=payload.answers,
        calculated_scores=calculated_results,
        current_stage=AssessmentStage.evaluated,
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    if payload.request_ai_interpretation:
        # تشغيل في الخلفية حتى لا يعطّل استجابة الطلب الرئيسي إن كان OpenAI بطيئاً
        background_tasks.add_task(_attach_ai_interpretation, assessment.id, calculated_results)

    return assessment


def _attach_ai_interpretation(assessment_id: int, scores: dict) -> None:
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        text = generate_interpretation(scores)
        assessment = db.get(UserAssessment, assessment_id)
        if assessment:
            assessment.ai_interpretation = text
            db.commit()
    finally:
        db.close()


@router.get("/{assessment_id}", response_model=AssessmentOut)
def get_assessment(
    assessment_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    assessment = db.get(UserAssessment, assessment_id)
    if not assessment or assessment.user_id != user.id:
        raise HTTPException(status_code=404, detail="التقييم غير موجود")
    return assessment


@router.get("", response_model=list[AssessmentOut])
def list_my_assessments(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return (
        db.query(UserAssessment)
        .filter(UserAssessment.user_id == user.id)
        .order_by(UserAssessment.created_at.desc())
        .all()
    )
