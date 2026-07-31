from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter(tags=["Health"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    """
    فحص حي يتحقق فعلياً من الاتصال بقاعدة البيانات، بدل الاكتفاء بإرجاع 200 ثابتة
    (وهو خطأ شائع يجعل الـ healthcheck عديم الفائدة عند تعطل قاعدة البيانات فقط).
    """
    db.execute(text("SELECT 1"))
    return {"status": "ok"}
