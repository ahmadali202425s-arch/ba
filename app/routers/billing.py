import logging

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import get_settings
from app.database import get_db
from app.models import Subscription, SubscriptionStatus, User

router = APIRouter(prefix="/api/billing", tags=["Billing"])
logger = logging.getLogger("coesis.billing")
settings = get_settings()

if settings.stripe_api_key:
    stripe.api_key = settings.stripe_api_key


@router.post("/create-checkout-session")
def create_checkout_session(price_id: str, user: User = Depends(get_current_user)):
    if not settings.stripe_api_key:
        raise HTTPException(status_code=503, detail="بوابة الدفع غير مفعّلة على هذا الخادم")
    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=user.email,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"https://{settings.domain}/billing/success",
            cancel_url=f"https://{settings.domain}/billing/cancel",
            client_reference_id=user.id,
        )
    except stripe.StripeError as exc:
        logger.error("فشل إنشاء جلسة الدفع: %s", exc)
        raise HTTPException(status_code=502, detail="تعذّر إنشاء جلسة الدفع")
    return {"checkout_url": session.url}


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """
    نقطة استقبال أحداث Stripe. يتم التحقق من التوقيع دائماً (stripe_webhook_secret)
    لمنع انتحال طلبات دفع مزيفة - هذه خطوة أمان إلزامية وليست اختيارية.
    """
    if not settings.stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook غير مهيأ")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, settings.stripe_webhook_secret)
    except (ValueError, stripe.SignatureVerificationError):
        raise HTTPException(status_code=400, detail="توقيع Webhook غير صالح")

    data = event["data"]["object"]

    if event["type"] == "checkout.session.completed":
        user_id = data.get("client_reference_id")
        if user_id:
            sub = db.query(Subscription).filter(Subscription.user_id == user_id).first()
            if not sub:
                sub = Subscription(user_id=user_id)
                db.add(sub)
            sub.stripe_customer_id = data.get("customer")
            sub.stripe_subscription_id = data.get("subscription")
            sub.status = SubscriptionStatus.active
            db.commit()

    elif event["type"] in ("customer.subscription.deleted", "customer.subscription.canceled"):
        sub = (
            db.query(Subscription)
            .filter(Subscription.stripe_subscription_id == data.get("id"))
            .first()
        )
        if sub:
            sub.status = SubscriptionStatus.canceled
            db.commit()

    return {"received": True}
