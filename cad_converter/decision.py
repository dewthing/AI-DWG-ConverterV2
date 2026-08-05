"""Automatic input analysis and processing decisions for drawing conversion.

The decision engine is deliberately inspectable.  It measures the uploaded page,
chooses suitable OpenCV restoration strategies, and records the reasons in the
conversion report.  A learned candidate ranker is applied later by
``FeedbackLearner``; these image-quality rules provide a safe cold start before
enough labelled drawings exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


@dataclass(slots=True)
class DrawingProfile:
    width: int
    height: int
    megapixels: float
    laplacian_variance: float
    contrast: float
    illumination_variation: float
    noise_level: float
    ink_ratio: float
    colour_ink_ratio: float
    quality_score: float
    recommended_iterations: int
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IterationDecision:
    iteration: int
    strategies: list[str]
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AutoDecisionEngine:
    """Profile a page and select complementary preprocessing candidates."""

    def analyse(self, image_bgr: np.ndarray) -> DrawingProfile:
        if image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
            raise ValueError("AutoDecisionEngine expects a BGR colour image.")

        height, width = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        contrast = float(gray.std())

        # A broad blur approximates the paper/background illumination.  Its
        # variation catches shadows and photographed drawings with uneven light.
        small = cv2.resize(
            gray,
            (min(width, 512), max(1, round(height * min(width, 512) / width))),
            interpolation=cv2.INTER_AREA,
        )
        kernel_size = _odd_at_most(max(15, min(small.shape[:2]) // 12), 51)
        background = cv2.GaussianBlur(small, (kernel_size, kernel_size), 0)
        illumination_variation = float(background.std())

        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        noise_level = float(np.mean(cv2.absdiff(gray, denoised)))

        _, ink = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        ink_ratio = float(np.count_nonzero(ink) / max(ink.size, 1))
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        coloured = (hsv[:, :, 1] > 45) & (hsv[:, :, 2] < 250)
        colour_ink_ratio = float(np.count_nonzero(coloured) / max(coloured.size, 1))

        flags: list[str] = []
        if laplacian_variance < 45.0:
            flags.append("blurred")
        if contrast < 30.0:
            flags.append("low_contrast")
        if illumination_variation > 24.0:
            flags.append("uneven_lighting")
        if noise_level > 6.5:
            flags.append("noisy")
        if colour_ink_ratio > 0.004:
            flags.append("coloured_ink")
        if ink_ratio < 0.0015:
            flags.append("very_sparse")
        elif ink_ratio > 0.30:
            flags.append("very_dense")

        sharpness_score = float(np.clip(laplacian_variance / 180.0, 0.0, 1.0))
        contrast_score = float(np.clip(contrast / 55.0, 0.0, 1.0))
        lighting_score = float(np.clip(1.0 - illumination_variation / 65.0, 0.0, 1.0))
        noise_score = float(np.clip(1.0 - noise_level / 18.0, 0.0, 1.0))
        quality_score = (
            0.35 * sharpness_score
            + 0.30 * contrast_score
            + 0.20 * lighting_score
            + 0.15 * noise_score
        )

        severe = {"blurred", "low_contrast", "uneven_lighting"}.intersection(flags)
        recommended_iterations = 3 if severe else (2 if flags else 1)
        return DrawingProfile(
            width=width,
            height=height,
            megapixels=round(width * height / 1_000_000.0, 3),
            laplacian_variance=round(laplacian_variance, 3),
            contrast=round(contrast, 3),
            illumination_variation=round(illumination_variation, 3),
            noise_level=round(noise_level, 3),
            ink_ratio=round(ink_ratio, 6),
            colour_ink_ratio=round(colour_ink_ratio, 6),
            quality_score=round(float(quality_score), 4),
            recommended_iterations=recommended_iterations,
            flags=flags,
        )

    def decide_iteration(
        self,
        profile: DrawingProfile,
        iteration: int,
        auto_mode: bool = True,
    ) -> IterationDecision:
        if not auto_mode:
            return IterationDecision(
                iteration=iteration + 1,
                strategies=[],
                reasons=["manual mode: evaluate every strategy available for this iteration"],
            )

        flags = set(profile.flags)
        strategies: list[str]
        reasons: list[str] = []
        if iteration == 0:
            strategies = ["otsu", "adaptive", "clahe_otsu"]
            reasons.append("start with stable document thresholds")
            if "coloured_ink" in flags:
                strategies.append("colour_ink")
                reasons.append("colour ink was detected")
            if "blurred" in flags or "very_sparse" in flags:
                strategies.append("edge_preserving")
                reasons.append("retain faint or blurred edges")
        elif iteration == 1:
            strategies = ["gamma_light_otsu", "gamma_dark_otsu", "adaptive_closed"]
            if "uneven_lighting" in flags or "low_contrast" in flags:
                strategies.append("background_normalized")
                reasons.append("correct low contrast or uneven paper lighting")
            if "noisy" in flags:
                strategies.append("bilateral_adaptive")
                reasons.append("suppress scan noise while preserving edges")
            if not reasons:
                reasons.append("try stronger tonal and morphology restoration")
        else:
            strategies = ["unsharp_otsu", "otsu_edges_hybrid", "blackhat_ink"]
            if "noisy" in flags or "blurred" in flags:
                strategies.append("bilateral_adaptive")
            reasons.append("final recovery pass for weak, broken, or blurred geometry")

        # Preserve order while avoiding duplicate work.
        strategies = list(dict.fromkeys(strategies))
        return IterationDecision(
            iteration=iteration + 1,
            strategies=strategies,
            reasons=reasons,
        )


def profile_warnings(profile: DrawingProfile) -> list[str]:
    """Create concise user-facing warnings for source limitations."""

    messages = {
        "blurred": "The source appears blurred; automatic recovery cannot recreate detail absent from the file.",
        "low_contrast": "The source has low contrast; enhanced threshold variants were enabled.",
        "uneven_lighting": "Uneven page lighting was detected; background normalization was enabled.",
        "noisy": "Scan noise was detected; edge-preserving denoising was enabled.",
        "very_sparse": "Very little drawing ink was detected; verify that the correct page was uploaded.",
        "very_dense": "The page is unusually dense; inspect the QA overlay for merged geometry.",
    }
    return [messages[flag] for flag in profile.flags if flag in messages]


def _odd_at_most(value: int, maximum: int) -> int:
    result = min(maximum, max(3, value))
    return result if result % 2 == 1 else result - 1
