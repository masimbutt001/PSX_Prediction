"""Financial domain sentiment analyzer with negation awareness and intensity scaling."""

import math
import re
from dataclasses import dataclass, field

POSITIVE_FINANCIAL_WORDS: set[str] = {
    "surge",
    "surges",
    "surged",
    "surging",
    "gain",
    "gains",
    "gained",
    "gaining",
    "profit",
    "profits",
    "profitable",
    "profitability",
    "growth",
    "grew",
    "grow",
    "jump",
    "jumps",
    "jumped",
    "rally",
    "rallied",
    "rallies",
    "record",
    "beat",
    "beats",
    "beating",
    "upgrade",
    "upgraded",
    "upgrades",
    "bullish",
    "boom",
    "booming",
    "dividend",
    "dividends",
    "bonus",
    "expansion",
    "expanded",
    "acquisition",
    "recovery",
    "recovers",
    "recovered",
    "optimism",
    "optimistic",
    "highest",
    "soar",
    "soared",
    "soars",
    "uptick",
    "outperform",
    "outperformed",
    "milestone",
    "success",
    "successful",
    "strengthen",
    "strengthened",
    "inflow",
    "inflows",
    "surplus",
    "discovery",
    "discover",
    "discovers",
    "discovered",
    "discovering",
}

NEGATIVE_FINANCIAL_WORDS: set[str] = {
    "plunge",
    "plunges",
    "plunged",
    "loss",
    "losses",
    "decline",
    "declines",
    "declined",
    "drop",
    "drops",
    "dropped",
    "fall",
    "falls",
    "fell",
    "slump",
    "slumps",
    "slumped",
    "deficit",
    "deficits",
    "default",
    "defaulted",
    "defaults",
    "warning",
    "warns",
    "warned",
    "inflation",
    "inflationary",
    "shutdown",
    "shutdowns",
    "downgrade",
    "downgraded",
    "downgrades",
    "bearish",
    "crisis",
    "crash",
    "crashed",
    "penalty",
    "penalties",
    "fined",
    "fine",
    "lowest",
    "downside",
    "underperform",
    "underperformed",
    "debt",
    "dispute",
    "conflict",
    "halt",
    "halted",
    "suspension",
    "suspended",
    "outflow",
    "outflows",
    "weakness",
    "weakened",
    "struggle",
    "struggling",
}

NEGATION_WORDS: set[str] = {
    "not",
    "no",
    "never",
    "failed",
    "fail",
    "fails",
    "without",
    "cannot",
    "hardly",
    "barely",
}

INTENSIFIER_WORDS: set[str] = {
    "record",
    "sharp",
    "sharply",
    "massive",
    "massively",
    "substantial",
    "substantially",
    "historic",
    "huge",
    "significantly",
    "significant",
    "steep",
    "steeply",
}


@dataclass(frozen=True)
class SentimentResult:
    """Quantitative evaluation of financial text sentiment."""

    score: float  # [-1.0, +1.0]
    label: str  # "POSITIVE", "NEGATIVE", "NEUTRAL"
    confidence: float  # [0.0, 1.0]
    positive_tokens: list[str] = field(default_factory=list)
    negative_tokens: list[str] = field(default_factory=list)


class FinancialSentimentAnalyzer:
    """Financial domain lexicon analyzer with 3-token negation and intensity scaling."""

    def __init__(self) -> None:
        self._word_regex = re.compile(r"\b[a-zA-Z]{2,}\b")

    def analyze(self, text: str) -> SentimentResult:
        """Calculate normalized financial sentiment score and classification.

        Args:
            text: Headline, article summary, or announcement body.

        Returns:
            SentimentResult with score, categorical label, and contributing tokens.
        """
        if not text or not text.strip():
            return SentimentResult(score=0.0, label="NEUTRAL", confidence=0.5)

        tokens = [w.lower() for w in self._word_regex.findall(text)]
        if not tokens:
            return SentimentResult(score=0.0, label="NEUTRAL", confidence=0.5)

        raw_score = 0.0
        pos_tokens: list[str] = []
        neg_tokens: list[str] = []

        for i, token in enumerate(tokens):
            # Check 3-token preceding lookback window for negation
            is_negated = False
            lookback_start = max(0, i - 3)
            for prev_idx in range(lookback_start, i):
                if tokens[prev_idx] in NEGATION_WORDS:
                    is_negated = True
                    break

            # Check 2-token preceding window for intensifier
            is_intensified = False
            lookback_intensifier = max(0, i - 2)
            for prev_idx in range(lookback_intensifier, i):
                if tokens[prev_idx] in INTENSIFIER_WORDS:
                    is_intensified = True
                    break

            multiplier = 1.5 if is_intensified else 1.0

            if token in POSITIVE_FINANCIAL_WORDS:
                if is_negated:
                    raw_score -= 1.0 * multiplier
                    neg_tokens.append(f"not_{token}")
                else:
                    raw_score += 1.0 * multiplier
                    pos_tokens.append(token)

            elif token in NEGATIVE_FINANCIAL_WORDS:
                if is_negated:
                    raw_score += 0.8 * multiplier  # 'not a loss' is mildly positive
                    pos_tokens.append(f"not_{token}")
                else:
                    raw_score -= 1.0 * multiplier
                    neg_tokens.append(token)

        # Smooth non-linear normalization into [-1.0, +1.0] using hyperbolic tangent
        normalized_score = math.tanh(raw_score / 2.5)
        normalized_score = round(normalized_score, 4)

        total_sentiment_words = len(pos_tokens) + len(neg_tokens)
        if total_sentiment_words == 0:
            confidence = 0.5
            label = "NEUTRAL"
        else:
            # Confidence rises with number of recognized emotional tokens
            confidence = min(0.6 + (total_sentiment_words * 0.1), 0.95)
            if normalized_score >= 0.05:
                label = "POSITIVE"
            elif normalized_score <= -0.05:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

        return SentimentResult(
            score=normalized_score,
            label=label,
            confidence=round(confidence, 2),
            positive_tokens=pos_tokens,
            negative_tokens=neg_tokens,
        )
