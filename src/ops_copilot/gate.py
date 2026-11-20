"""Confidence gate: refuse before the LLM is ever called when retrieval is weak."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    confidence: float
    threshold: float


def check(confidence: float, threshold: float) -> GateDecision:
    """confidence is the max dense cosine similarity among candidates; pass iff >= threshold."""
    return GateDecision(confidence >= threshold, confidence, threshold)
