"""
محرك التنبؤ بالمخاطر التنظيمية (v6.1)
--------------------------------------
منقول ومُدمج من الطبقة التنبؤية التي طوّرها الباحث. يقدّر احتمال وجود فجوة
في "الترجمة القيمية" (Value Translation) وأنسنة التكنولوجيا داخل منظمة ما،
اعتماداً على أربعة مؤشرات كمية.

ملاحظة منهجية: النموذج يُؤسَّس ابتدائياً على بيانات تدريب اصطناعية صغيرة
(Bootstrap, 4 عيّنات فقط) لضمان عمل الـ API فوراً دون انتظار تجميع بيانات
حقيقية كافية. هذا مناسب للتجريب والعرض التوضيحي فقط ولا يمثّل نموذجاً
مُتعلَّماً إحصائياً بأي معنى - أي استجابة تعتمد عليه تحمل `is_bootstrap: true`
بشكل صريح. يجب استدعاء `retrain()` (عبر `/api/ai/retrain`) على بيانات استبيان
حقيقية (10 عيّنات على الأقل) قبل الاعتماد على المخرجات في أي قرار فعلي.

جديد في v6.1: استمرار (persistence) حقيقي عبر joblib - النموذج المُعاد
تدريبه يُحفَظ على القرص ويُحمَّل عند إقلاع كل عملية uvicorn بدل إعادة بنائه
من الصفر في كل عامل (worker) بمعزل عن العمّال الآخرين.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("coesis.ai_engine")

# ترتيب المؤشرات الأربعة الذي يعتمده النموذج - أي مستدعٍ يجب أن يمرر القيم
# بنفس هذا الترتيب (راجع app/routers/ai_insights.py)
DEFAULT_FEATURE_KEYS: Sequence[str] = (
    "value_translation_score",
    "humanizing_tech_score",
    "human_smart_compatibility",
    "operational_safety_score",
)

# الحد الأدنى لحجم العيّنة المقبول لإعادة تدريب حقيقية - أقل من هذا يُعيد
# إنتاج نفس مشكلة "تحفظ 4 نقاط بدل التعلّم" التي بدأنا منها أصلاً.
MIN_RETRAIN_SAMPLES = 10


class AdvancedPredictiveEngine:
    """محرك تعلّم آلي (Random Forest) لقياس جودة الترجمة القيمية وأنسنة
    التكنولوجيا، والتنبؤ باحتمال وجود مخاطر تنظيمية (إنهاك/فجوة توافق)."""

    def __init__(self, feature_keys: Sequence[str] = DEFAULT_FEATURE_KEYS):
        self.feature_keys = list(feature_keys)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        self.is_bootstrap = True
        self.trained_at: str = datetime.now(timezone.utc).isoformat()
        self.train_sample_size = 0
        self.train_accuracy: Optional[float] = None
        self._fit_bootstrap()

    # ---------- تدريب ----------

    def _fit_bootstrap(self) -> None:
        """بيانات تأسيسية لتدريب أولي إلى حين توفر بيانات استبيان حقيقية كافية.
        هذه ليست بديلاً عن تدريب حقيقي - فقط تضمن استجابة صالحة شكلياً."""
        x_dummy = pd.DataFrame(
            [
                [1.2, 1.5, 2.0, 4.0],
                [4.5, 4.8, 4.5, 1.5],
                [3.0, 3.0, 3.0, 3.0],
                [1.0, 1.1, 1.5, 4.5],
            ],
            columns=self.feature_keys,
        )
        y_dummy = np.array([1, 0, 0, 1])
        self.pipeline.fit(x_dummy, y_dummy)
        self._classes = set(self.pipeline.named_steps["classifier"].classes_.tolist())
        self.is_bootstrap = True
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.train_sample_size = len(x_dummy)
        self.train_accuracy = None

    def retrain(self, x: np.ndarray, y: np.ndarray) -> None:
        """إعادة تدريب النموذج على بيانات استبيان حقيقية مُصنَّفة (يستبدل بيانات
        bootstrap). يرفض العيّنات الصغيرة جداً بدل قبولها بصمت والوقوع في نفس
        مشكلة "الحفظ عن ظهر قلب" التي كانت في بيانات bootstrap الأصلية."""
        n = len(y)
        if n < MIN_RETRAIN_SAMPLES:
            raise ValueError(
                f"عدد العيّنات ({n}) أقل من الحد الأدنى المطلوب لإعادة تدريب موثوقة "
                f"({MIN_RETRAIN_SAMPLES}). هذا ليس قيداً تعسفياً: تدريب Random Forest "
                "على عيّنات قليلة جداً ينتج نموذجاً يحفظ الأمثلة بدل أن يتعلّم علاقة "
                "إحصائية حقيقية - وهو بالضبط الخلل الذي يهدف هذا الحد إلى تجنّبه."
            )
        if len(set(y.tolist())) < 2:
            raise ValueError("تدريب مصنِّف Random Forest يتطلب فئتين على الأقل (0 و1) في y.")

        # نلتزم بأسماء الأعمدة نفسها المستخدمة في _fit_bootstrap لتفادي تحذير
        # sklearn حول "X has feature names, but StandardScaler was fitted
        # without feature names" - وهو أيضاً أكثر أماناً لو تغيّر ترتيب الأعمدة لاحقاً.
        x = pd.DataFrame(x, columns=self.feature_keys)

        # فاصل تدريب/تحقق بسيط لتقدير دقة تقريبية (ليس بديلاً عن تحقق متقاطع كامل،
        # لكنه أفضل من عدم عرض أي مؤشر أداء على الإطلاق)
        train_accuracy: Optional[float]
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            x_train, x_val, y_train, y_val = train_test_split(
                x, y, test_size=0.2, random_state=42, stratify=y
            )
            self.pipeline.fit(x_train, y_train)
            train_accuracy = float(accuracy_score(y_val, self.pipeline.predict(x_val)))
            # التدريب النهائي على كامل البيانات بعد تقدير الدقة التقريبية
            self.pipeline.fit(x, y)
        except ValueError:
            # عيّنة صغيرة جداً لتقسيم طبقي (stratify) - ندرّب على كامل البيانات بلا تقدير دقة
            self.pipeline.fit(x, y)
            train_accuracy = None

        self._classes = set(self.pipeline.named_steps["classifier"].classes_.tolist())
        self.is_bootstrap = False
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.train_sample_size = n
        self.train_accuracy = train_accuracy

    # ---------- استمرار (persistence) ----------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(
            "تم حفظ نموذج التنبؤ على %s (bootstrap=%s, sample_size=%s)",
            path, self.is_bootstrap, self.train_sample_size,
        )

    @classmethod
    def load_or_create(cls, path: str | Path) -> "AdvancedPredictiveEngine":
        """يحمّل نموذجاً محفوظاً مسبقاً على القرص إن وُجد وكان صالحاً؛ وإلا ينشئ
        نموذج bootstrap جديداً ويحفظه فوراً حتى تشترك فيه بقية عمليات uvicorn."""
        path = Path(path)
        if path.exists():
            try:
                engine = joblib.load(path)
                if isinstance(engine, cls) and hasattr(engine, "pipeline"):
                    logger.info(
                        "تم تحميل نموذج تنبؤ محفوظ من %s (bootstrap=%s)",
                        path, getattr(engine, "is_bootstrap", True),
                    )
                    return engine
                logger.warning("الملف المحفوظ في %s ليس نموذج AdvancedPredictiveEngine صالحاً - يُتجاهَل.", path)
            except Exception:
                logger.exception("تعذّر تحميل النموذج المحفوظ من %s - سيُنشأ نموذج bootstrap جديد.", path)

        engine = cls()
        try:
            engine.save(path)
        except OSError:
            logger.exception("تعذّر حفظ نموذج bootstrap الجديد على %s (هل المجلد قابل للكتابة؟)", path)
        return engine

    # ---------- استدلال ----------

    def metadata(self) -> Dict[str, Any]:
        return {
            "is_bootstrap": self.is_bootstrap,
            "trained_at": self.trained_at,
            "train_sample_size": self.train_sample_size,
            "train_accuracy": self.train_accuracy,
        }

    def evaluate_organizational_risk(self, metrics_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics_records:
            return {"status": "error", "message": "لا توجد سجلات كافية للتحليل الإحصائي."}

        df = pd.DataFrame(metrics_records)
        missing_keys = [k for k in self.feature_keys if k not in df.columns]
        for k in missing_keys:
            df[k] = 3.0  # قيمة محايدة افتراضية لأي مؤشر غائب عن السجلات
        features = df[self.feature_keys].fillna(3.0)

        # حماية: إن كانت الفئة 1 (خطر) غير موجودة أصلاً في تدريب النموذج الحالي،
        # predict_proba تعيد عموداً واحداً فقط - نتجنب هنا IndexError
        proba = self.pipeline.predict_proba(features)
        risk_col_index = list(self.pipeline.named_steps["classifier"].classes_).index(1) if 1 in self._classes else None
        probabilities = proba[:, risk_col_index] if risk_col_index is not None else np.zeros(len(features))
        mean_prob = float(np.mean(probabilities))

        risk_tier = (
            "حرج ومرتفع (فجوة في الترجمة القيمية وأنسنة التكنولوجيا)"
            if mean_prob > 0.65
            else ("متوسط ويحتاج توجيه إداري" if mean_prob > 0.35 else "مستقر آمن وتوافق عالي")
        )

        return {
            "status": "success",
            "model_type": "RandomForestClassifier (Ensemble Learning - v6.1)",
            "sample_size": len(features),
            "used_default_for_missing_indicators": missing_keys or None,
            "mean_risk_probability": round(mean_prob, 3),
            "risk_tier": risk_tier,
            "actionable_insight": (
                "يوصى بتفعيل بروتوكول تحويل القيم التنظيمية إلى سياسات إدارية واضحة "
                "وتأطير أدوات الذكاء الاصطناعي."
            )
            if mean_prob > 0.35
            else "البيئة الثقافية والتنظيمية تظهر ترجمة قيمية مستقرة ومتوافقة.",
            "model_metadata": self.metadata(),
            "reliability_warning": (
                "هذا النموذج لا يزال على بيانات bootstrap توضيحية (4 عيّنات اصطناعية) "
                "ولم يُعَد تدريبه على بيانات استبيان حقيقية بعد. النتائج هنا لأغراض العرض "
                "التوضيحي فقط ولا ينبغي الاعتماد عليها في قرارات فعلية."
            )
            if self.is_bootstrap
            else None,
        }
