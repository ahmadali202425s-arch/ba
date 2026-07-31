"""
نماذج قاعدة البيانات.
يحافظ هذا الملف على فكرة الجداول المرنة (scales / dimensions / questions) من التصميم
الأصلي حتى يمكن إضافة مقاييس جديدة (كمقياس القيم الإسلامية أو مقياس التوافق البشري
مع الذكاء الاصطناعي) دون أي تعديل في الشيفرة البرمجية، مع إضافة نماذج المستخدمين
والاشتراكات الضرورية لتشغيل منصة حقيقية.
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class UserRole(str, enum.Enum):
    respondent = "respondent"       # مستخدم يجيب عن المقاييس
    researcher = "researcher"       # يضيف مقاييسه الخاصة
    admin = "admin"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.respondent)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    assessments: Mapped[list["UserAssessment"]] = relationship(back_populates="user")
    subscription: Mapped["Subscription | None"] = relationship(back_populates="user", uselist=False)


class Scale(Base):
    """المقياس - أساسي (AI Baseline) أو مخصص يضيفه باحث/مسؤول."""

    __tablename__ = "scales"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    creator_type: Mapped[str] = mapped_column(String(50), default="system")  # ai_baseline | admin_custom | researcher
    created_by_id: Mapped[str | None] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    dimensions: Mapped[list["ScaleDimension"]] = relationship(
        back_populates="scale", cascade="all, delete-orphan"
    )
    assessments: Mapped[list["UserAssessment"]] = relationship(back_populates="scale")


class ScaleDimension(Base):
    __tablename__ = "scale_dimensions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scale_id: Mapped[int] = mapped_column(ForeignKey("scales.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    weight_multiplier: Mapped[float] = mapped_column(Float, default=1.0)

    scale: Mapped["Scale"] = relationship(back_populates="dimensions")
    questions: Mapped[list["ScaleQuestion"]] = relationship(
        back_populates="dimension", cascade="all, delete-orphan"
    )


class ScaleQuestion(Base):
    __tablename__ = "scale_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scale_id: Mapped[int] = mapped_column(ForeignKey("scales.id", ondelete="CASCADE"))
    dimension_id: Mapped[int] = mapped_column(ForeignKey("scale_dimensions.id", ondelete="CASCADE"))
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    likert_scale_type: Mapped[str] = mapped_column(String(50), default="standard_5")
    # الحد الأدنى/الأقصى الفعليين لقيمة ليكرت حتى يمكن التحقق من صحة الإجابات ديناميكياً
    likert_min: Mapped[int] = mapped_column(Integer, default=1)
    likert_max: Mapped[int] = mapped_column(Integer, default=5)
    is_reverse_scored: Mapped[bool] = mapped_column(Boolean, default=False)

    dimension: Mapped["ScaleDimension"] = relationship(back_populates="questions")


class AssessmentStage(str, enum.Enum):
    diagnosed = "diagnosed"
    analyzed = "analyzed"
    evaluated = "evaluated"
    intervention_active = "intervention_active"
    sustainable_accompanying = "sustainable_accompanying"


class UserAssessment(Base):
    __tablename__ = "user_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    scale_id: Mapped[int] = mapped_column(ForeignKey("scales.id"))
    raw_answers: Mapped[dict] = mapped_column(JSONB, nullable=False)
    calculated_scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    ai_interpretation: Mapped[str | None] = mapped_column(Text, nullable=True)
    current_stage: Mapped[AssessmentStage] = mapped_column(
        Enum(AssessmentStage), default=AssessmentStage.diagnosed
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="assessments")
    scale: Mapped["Scale"] = relationship(back_populates="assessments")


class SubscriptionStatus(str, enum.Enum):
    trialing = "trialing"
    active = "active"
    past_due = "past_due"
    canceled = "canceled"


class Subscription(Base):
    """اشتراك المستخدم عبر Stripe (بوابة الدفع)."""

    __tablename__ = "subscriptions"
    __table_args__ = (UniqueConstraint("user_id", name="uq_subscription_user"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"), nullable=False)
    stripe_customer_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[SubscriptionStatus] = mapped_column(
        Enum(SubscriptionStatus), default=SubscriptionStatus.trialing
    )
    current_period_end: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="subscription")
