"""
محرك المرافقة المستدامة (Sustainable Accompanying Engine).

عملية خلفية مستقلة (خدمة learning_worker في docker-compose) تفحص دورياً التقييمات
التي وصلت لمرحلة "intervention_active" وتحوّلها إلى "sustainable_accompanying"
بعد توليد متابعة مبسطة عبر المساعد الذكي. مصممة لتتحمل انقطاع الاتصال بقاعدة
البيانات أو Redis دون أن تتوقف العملية بأكملها (إعادة محاولة بدل الانهيار).
"""
import logging
import signal
import time

from sqlalchemy import select
from tenacity import retry, stop_after_attempt, wait_fixed

from app.database import SessionLocal
from app.models import AssessmentStage, UserAssessment
from app.services.ai_assistant import generate_interpretation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("coesis.learning_worker")

POLL_INTERVAL_SECONDS = 60
_shutdown = False


def _handle_shutdown(signum, frame):
    global _shutdown
    logger.info("إشارة إيقاف مستلمة (%s) - سيتم إنهاء الدورة الحالية ثم التوقف", signum)
    _shutdown = True


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def _process_pending_batch() -> int:
    db = SessionLocal()
    processed = 0
    try:
        pending = (
            db.execute(
                select(UserAssessment)
                .where(UserAssessment.current_stage == AssessmentStage.intervention_active)
                .limit(50)
            )
            .scalars()
            .all()
        )
        for assessment in pending:
            interpretation = generate_interpretation(assessment.calculated_scores)
            if interpretation:
                assessment.ai_interpretation = interpretation
            assessment.current_stage = AssessmentStage.sustainable_accompanying
            processed += 1
        db.commit()
    finally:
        db.close()
    return processed


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)
    logger.info("بدء محرك المرافقة المستدامة")

    while not _shutdown:
        try:
            count = _process_pending_batch()
            if count:
                logger.info("تمت معالجة %d تقييماً", count)
        except Exception:
            logger.exception("فشلت دورة المعالجة بعد إعادة المحاولات - سيُعاد بعد الفاصل الزمني")
        for _ in range(POLL_INTERVAL_SECONDS):
            if _shutdown:
                break
            time.sleep(1)

    logger.info("توقف محرك المرافقة المستدامة بشكل نظيف")


if __name__ == "__main__":
    main()
