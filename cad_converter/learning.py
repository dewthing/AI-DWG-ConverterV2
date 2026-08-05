"""Local feedback learning for better candidate selection over time.

This is intentionally human-in-the-loop learning. It records the user's quality
score and trains only on those explicit corrections; it never silently changes
or fabricates engineering geometry.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import CandidateMetrics


FEATURE_NAMES = (
    "geometric_score",
    "tolerant_f1",
    "exact_iou",
    "ocr_score",
    "ink_ratio",
    "line_count",
    "circle_count",
    "polyline_count",
    "text_count",
    "fragmentation",
)

MIN_TRAINING_RECORDS = 5
MIN_NEURAL_RECORDS = 30


@dataclass(slots=True)
class TrainingSummary:
    trained: bool
    record_count: int
    message: str
    model_kind: str | None = None


class FeedbackLearner:
    """Persist feedback and optionally train a compact local ranking model."""

    def __init__(self, feedback_path: str | Path) -> None:
        self.feedback_path = Path(feedback_path)
        self.model_path = self.feedback_path.with_suffix(".joblib")
        self._model: object | None | bool = False

    def predict(self, metrics: CandidateMetrics) -> float | None:
        model = self._load_model()
        if model is None:
            return None
        vector = self._feature_vector(metrics).reshape(1, -1)
        return float(np.clip(model.predict(vector)[0], 0.0, 1.0))

    def record_feedback(
        self,
        source_name: str,
        candidate_name: str,
        metrics: CandidateMetrics,
        score_percent: float,
        accepted: bool,
        note: str = "",
    ) -> TrainingSummary:
        score = float(np.clip(score_percent / 100.0, 0.0, 1.0))
        if accepted:
            score = max(score, 0.75)

        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "source_name": source_name,
            "candidate_name": candidate_name,
            "accepted": bool(accepted),
            "note": note.strip(),
            "score": score,
            "features": asdict(metrics),
        }
        with self.feedback_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._model = False
        return self.train()

    def train(self) -> TrainingSummary:
        records = self._records()
        valid = [
            record
            for record in records
            if isinstance(record.get("features"), dict) and "score" in record
        ]
        if len(valid) < MIN_TRAINING_RECORDS:
            return TrainingSummary(
                trained=False,
                record_count=len(valid),
                message=(
                    f"Saved feedback. {MIN_TRAINING_RECORDS - len(valid)} more labelled conversion(s) "
                    "are needed before the local ranking model trains."
                ),
            )

        features = np.asarray(
            [
                [
                    float(record["features"].get(name, 0.0))
                    for name in FEATURE_NAMES
                ]
                for record in valid
            ],
            dtype=np.float64,
        )
        scores = np.asarray([float(record["score"]) for record in valid], dtype=np.float64)
        if len(valid) >= MIN_NEURAL_RECORDS:
            model: object = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "network",
                        MLPRegressor(
                            hidden_layer_sizes=(64, 32, 16),
                            activation="relu",
                            solver="adam",
                            alpha=0.002,
                            max_iter=1200,
                            random_state=42,
                        ),
                    ),
                ]
            )
            model_kind = "neural_mlp"
        else:
            model = RandomForestRegressor(
                n_estimators=180,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )
            model_kind = "random_forest"
        model.fit(features, scores)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "feature_names": FEATURE_NAMES,
                "model": model,
                "model_kind": model_kind,
                "record_count": len(valid),
            },
            self.model_path,
        )
        self._model = model
        return TrainingSummary(
            trained=True,
            record_count=len(valid),
            message=(
                f"Saved feedback and trained the local {model_kind} candidate-ranking model "
                f"from {len(valid)} labelled conversion(s)."
            ),
            model_kind=model_kind,
        )

    def _load_model(self) -> object | None:
        if self._model is not False:
            return self._model
        if not self.model_path.exists():
            self._model = None
            return None
        try:
            payload = joblib.load(self.model_path)
            model = payload.get("model")
            names = tuple(payload.get("feature_names", ()))
            if names != FEATURE_NAMES or not callable(getattr(model, "predict", None)):
                self._model = None
            else:
                self._model = model
        except (OSError, ValueError, TypeError):
            self._model = None
        return self._model

    def _records(self) -> list[dict[str, Any]]:
        if not self.feedback_path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.feedback_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    @staticmethod
    def _feature_vector(metrics: CandidateMetrics) -> np.ndarray:
        raw = asdict(metrics)
        return np.asarray(
            [float(raw.get(name, 0.0)) for name in FEATURE_NAMES],
            dtype=np.float64,
        )
