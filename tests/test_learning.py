from __future__ import annotations

from cad_converter.learning import FeedbackLearner
from cad_converter.models import CandidateMetrics


def _metrics(offset: float) -> CandidateMetrics:
    return CandidateMetrics(
        geometric_score=0.65 + offset,
        tolerant_f1=0.70 + offset,
        exact_iou=0.45 + offset,
        ocr_score=0.80,
        learned_score=0.65,
        final_score=0.67,
        ink_ratio=0.08,
        line_count=20,
        circle_count=2,
        polyline_count=4,
        text_count=3,
        fragmentation=0.12,
    )


def test_feedback_trains_and_reloads_local_ranker(tmp_path):
    feedback_path = tmp_path / "feedback.jsonl"
    learner = FeedbackLearner(feedback_path)
    summary = None
    for index in range(5):
        summary = learner.record_feedback(
            source_name=f"drawing-{index}.png",
            candidate_name="adaptive",
            metrics=_metrics(index * 0.01),
            score_percent=70 + index * 4,
            accepted=True,
        )

    assert summary is not None
    assert summary.trained
    assert summary.model_kind == "random_forest"
    assert feedback_path.with_suffix(".joblib").exists()

    reloaded = FeedbackLearner(feedback_path)
    prediction = reloaded.predict(_metrics(0.02))
    assert prediction is not None
    assert 0.0 <= prediction <= 1.0
