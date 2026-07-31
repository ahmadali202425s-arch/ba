"""المخطط الأولي لقاعدة البيانات

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-31
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None

user_role = postgresql.ENUM("respondent", "researcher", "admin", name="userrole")
assessment_stage = postgresql.ENUM(
    "diagnosed", "analyzed", "evaluated", "intervention_active", "sustainable_accompanying",
    name="assessmentstage",
)
subscription_status = postgresql.ENUM(
    "trialing", "active", "past_due", "canceled", name="subscriptionstatus"
)


def upgrade() -> None:
    bind = op.get_bind()
    user_role.create(bind, checkfirst=True)
    assessment_stage.create(bind, checkfirst=True)
    subscription_status.create(bind, checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=False), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=True),
        sa.Column("role", user_role, nullable=False, server_default="respondent"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "scales",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("creator_type", sa.String(50), nullable=False, server_default="system"),
        sa.Column("created_by_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "scale_dimensions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("scale_id", sa.Integer, sa.ForeignKey("scales.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("weight_multiplier", sa.Float, nullable=False, server_default="1.0"),
    )

    op.create_table(
        "scale_questions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("scale_id", sa.Integer, sa.ForeignKey("scales.id", ondelete="CASCADE"), nullable=False),
        sa.Column(
            "dimension_id", sa.Integer, sa.ForeignKey("scale_dimensions.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("likert_scale_type", sa.String(50), nullable=False, server_default="standard_5"),
        sa.Column("likert_min", sa.Integer, nullable=False, server_default="1"),
        sa.Column("likert_max", sa.Integer, nullable=False, server_default="5"),
        sa.Column("is_reverse_scored", sa.Boolean, nullable=False, server_default=sa.false()),
    )

    op.create_table(
        "user_assessments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("scale_id", sa.Integer, sa.ForeignKey("scales.id"), nullable=False),
        sa.Column("raw_answers", postgresql.JSONB, nullable=False),
        sa.Column("calculated_scores", postgresql.JSONB, nullable=False),
        sa.Column("ai_interpretation", sa.Text, nullable=True),
        sa.Column("current_stage", assessment_stage, nullable=False, server_default="diagnosed"),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "subscriptions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=False), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(255), nullable=True),
        sa.Column("status", subscription_status, nullable=False, server_default="trialing"),
        sa.Column("current_period_end", sa.DateTime, nullable=True),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("user_id", name="uq_subscription_user"),
    )


def downgrade() -> None:
    op.drop_table("subscriptions")
    op.drop_table("user_assessments")
    op.drop_table("scale_questions")
    op.drop_table("scale_dimensions")
    op.drop_table("scales")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

    bind = op.get_bind()
    subscription_status.drop(bind, checkfirst=True)
    assessment_stage.drop(bind, checkfirst=True)
    user_role.drop(bind, checkfirst=True)
