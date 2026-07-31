"""
اختبارات وحدة لطبقة الذكاء الاصطناعي التنبؤي (v6.0) - app/ai_engine.
لا تتطلب قاعدة بيانات؛ يمكن تشغيلها مباشرة عبر: pytest backend/tests -q
"""

import numpy as np
import pytest

from app.ai_engine.nlp_sentiment import deep_semantic_cognitive_load_analysis
from app.ai_engine.predictive_models import AdvancedPredictiveEngine


def test_semantic_analysis_rejects_empty_batch():
    result = deep_semantic_cognitive_load_analysis([])
    assert result["status"] == "error"


def test_semantic_analysis_ignores_blank_texts():
    result = deep_semantic_cognitive_load_analysis(["   ", "", "أنا مرتاح وسعيد بالعمل"])
    assert result["status"] == "success"
    assert result["sample_size"] == 1


def test_semantic_analysis_flags_high_cognitive_load():
    result = deep_semantic_cognitive_load_analysis(
        ["أشعر بإرهاق وضغط واحتراق وظيفي شديد، الوضع مستحيل ومعقد"]
    )
    assert result["details"][0]["sentiment"] in {"سلبي", "محايد", "إيجابي"}
    assert result["details"][0]["flagged_markers"]["high_load"] > 0


def test_semantic_analysis_positive_text():
    result = deep_semantic_cognitive_load_analysis(["كل شيء ممتاز وواضح ومنظم، أشعر برضا كبير"])
    assert result["details"][0]["sentiment"] == "إيجابي"


def test_predictive_engine_handles_empty_records():
    engine = AdvancedPredictiveEngine()
    result = engine.evaluate_organizational_risk([])
    assert result["status"] == "error"


def test_predictive_engine_returns_risk_tier_for_stable_scores():
    engine = AdvancedPredictiveEngine()
    records = [
        {
            "value_translation_score": 4.5,
            "humanizing_tech_score": 4.6,
            "human_smart_compatibility": 4.4,
            "operational_safety_score": 1.5,
        }
    ]
    result = engine.evaluate_organizational_risk(records)
    assert result["status"] == "success"
    assert "risk_tier" in result
    assert 0.0 <= result["mean_risk_probability"] <= 1.0


def test_predictive_engine_defaults_missing_indicators():
    engine = AdvancedPredictiveEngine()
    # سجل ناقص عمداً - يجب ألا يفشل الاستدعاء، بل يستخدم قيمة محايدة (3.0)
    result = engine.evaluate_organizational_risk([{"value_translation_score": 2.0}])
    assert result["status"] == "success"
    assert result["used_default_for_missing_indicators"]


def test_new_engine_is_flagged_as_bootstrap():
    engine = AdvancedPredictiveEngine()
    result = engine.evaluate_organizational_risk(
        [{"value_translation_score": 3.0, "humanizing_tech_score": 3.0,
          "human_smart_compatibility": 3.0, "operational_safety_score": 3.0}]
    )
    assert result["model_metadata"]["is_bootstrap"] is True
    assert result["reliability_warning"] is not None


def test_retrain_rejects_samples_below_minimum():
    engine = AdvancedPredictiveEngine()
    x = np.array([[3.0, 3.0, 3.0, 3.0]] * 5)
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError):
        engine.retrain(x, y)


def test_retrain_rejects_single_class():
    engine = AdvancedPredictiveEngine()
    x = np.array([[3.0, 3.0, 3.0, 3.0]] * 12)
    y = np.zeros(12)
    with pytest.raises(ValueError):
        engine.retrain(x, y)


def test_retrain_succeeds_and_clears_bootstrap_flag():
    engine = AdvancedPredictiveEngine()
    rng = np.random.default_rng(42)
    x = rng.uniform(1, 5, size=(20, 4))
    y = np.array([1 if row[3] < 3 else 0 for row in x])
    engine.retrain(x, y)
    assert engine.is_bootstrap is False
    assert engine.train_sample_size == 20

    result = engine.evaluate_organizational_risk(
        [{"value_translation_score": 3.0, "humanizing_tech_score": 3.0,
          "human_smart_compatibility": 3.0, "operational_safety_score": 3.0}]
    )
    assert result["model_metadata"]["is_bootstrap"] is False
    assert result["reliability_warning"] is None


def test_save_and_load_round_trip_preserves_trained_state(tmp_path):
    engine = AdvancedPredictiveEngine()
    rng = np.random.default_rng(7)
    x = rng.uniform(1, 5, size=(15, 4))
    y = np.array([1 if row[3] < 3 else 0 for row in x])
    engine.retrain(x, y)

    model_path = tmp_path / "model_store" / "predictive_model.joblib"
    engine.save(model_path)
    assert model_path.exists()

    loaded = AdvancedPredictiveEngine.load_or_create(model_path)
    assert loaded.is_bootstrap is False
    assert loaded.train_sample_size == 15


def test_load_or_create_falls_back_to_bootstrap_when_missing(tmp_path):
    model_path = tmp_path / "does_not_exist" / "predictive_model.joblib"
    engine = AdvancedPredictiveEngine.load_or_create(model_path)
    assert engine.is_bootstrap is True
    # يجب أن يحفظ نموذج bootstrap الجديد فوراً حتى تشترك فيه بقية العمليات
    assert model_path.exists()
