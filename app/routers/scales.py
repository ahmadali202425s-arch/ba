from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, selectinload

from app.auth import get_current_user, require_role
from app.database import get_db
from app.models import Scale, ScaleDimension, ScaleQuestion, User
from app.schemas import ScaleCreate, ScaleOut

router = APIRouter(prefix="/api/scales", tags=["Scales CMS"])


@router.get("", response_model=list[ScaleOut])
def list_scales(active_only: bool = True, db: Session = Depends(get_db)):
    query = db.query(Scale).options(
        selectinload(Scale.dimensions).selectinload(ScaleDimension.questions)
    )
    if active_only:
        query = query.filter(Scale.is_active.is_(True))
    return query.order_by(Scale.id).all()


@router.get("/{scale_id}", response_model=ScaleOut)
def get_scale(scale_id: int, db: Session = Depends(get_db)):
    scale = (
        db.query(Scale)
        .options(selectinload(Scale.dimensions).selectinload(ScaleDimension.questions))
        .filter(Scale.id == scale_id)
        .first()
    )
    if not scale:
        raise HTTPException(status_code=404, detail="المقياس غير موجود")
    return scale


@router.post("", response_model=ScaleOut, status_code=201)
def create_scale(
    payload: ScaleCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin", "researcher")),
):
    """
    إضافة مقياس مخصص (مثل مقياس القيم الإسلامية أو مقياس التوافق البشري-الذكاء
    الاصطناعي) دون أي تعديل في الشيفرة البرمجية - هذا هو جوهر الـ Custom Scales CMS.
    """
    creator_type = "admin_custom" if user.role == "admin" else "researcher"
    scale = Scale(
        title=payload.title,
        description=payload.description,
        creator_type=creator_type,
        created_by_id=user.id,
    )
    db.add(scale)
    db.flush()  # للحصول على scale.id قبل الالتزام النهائي

    dimension_by_name: dict[str, ScaleDimension] = {}
    for dim in payload.dimensions:
        row = ScaleDimension(scale_id=scale.id, name=dim.name, weight_multiplier=dim.weight_multiplier)
        db.add(row)
        db.flush()
        dimension_by_name[dim.name] = row

    for q in payload.questions:
        db.add(
            ScaleQuestion(
                scale_id=scale.id,
                dimension_id=dimension_by_name[q.dimension_name].id,
                question_text=q.question_text,
                likert_scale_type=q.likert_scale_type,
                likert_min=q.likert_min,
                likert_max=q.likert_max,
                is_reverse_scored=q.is_reverse_scored,
            )
        )

    db.commit()
    db.refresh(scale)
    return scale


@router.patch("/{scale_id}/deactivate", response_model=ScaleOut)
def deactivate_scale(
    scale_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(require_role("admin")),
):
    scale = db.get(Scale, scale_id)
    if not scale:
        raise HTTPException(status_code=404, detail="المقياس غير موجود")
    scale.is_active = False
    db.commit()
    db.refresh(scale)
    return scale
