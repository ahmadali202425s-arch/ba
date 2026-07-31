from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.database import get_db
from app.models import User
from app.schemas import LoginRequest, RefreshRequest, Token, UserCreate, UserOut

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="البريد الإلكتروني مسجّل مسبقاً")
    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="البريد الإلكتروني أو كلمة المرور غير صحيحة")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="الحساب معطّل")
    return Token(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=Token)
def refresh(payload_in: RefreshRequest, db: Session = Depends(get_db)):
    # عبر جسم JSON بدل query parameter: توكن كهذا في الـ query string يتسرّب إلى
    # سجلات وصول nginx وتاريخ المتصفح ووسطاء الشبكة - راجع نقطة المراجعة السابقة.
    payload = decode_token(payload_in.refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="رمز التحديث غير صالح")
    user = db.get(User, payload.get("sub"))
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="المستخدم غير موجود أو معطّل")
    return Token(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
