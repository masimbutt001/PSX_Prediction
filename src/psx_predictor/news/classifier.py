"""Financial event category classifier for news headlines and company announcements."""

import re
from enum import Enum


class EventCategory(str, Enum):
    """Categorical classification of corporate and macroeconomic market events."""

    EARNINGS = "EARNINGS"
    DIVIDEND = "DIVIDEND"
    BONUS_ISSUE = "BONUS_ISSUE"
    DISCOVERY = "DISCOVERY"
    PRODUCTION_CHANGE = "PRODUCTION_CHANGE"
    REGULATORY = "REGULATORY"
    MACRO_INTEREST_RATE = "MACRO_INTEREST_RATE"
    OTHER = "OTHER"


CATEGORY_PATTERNS: dict[EventCategory, list[str]] = {
    EventCategory.EARNINGS: [
        r"\bfinancial\s+results?\b",
        r"\bquarterly\s+results?\b",
        r"\bhalf[- ]yearly\s+results?\b",
        r"\bannual\s+results?\b",
        r"\bprofit\s+after\s+tax\b",
        r"\bnet\s+profit\b",
        r"\bearnings\s+per\s+share\b",
        r"\beps\b",
        r"\bfinancial\s+statements?\b",
        r"\brevenue\s+(?:surge|growth|drop|decline|jumps?)\b",
        r"\bprofit\s+(?:surges?|plunges?|jumps?|falls?|gains?)\b",
        r"\bboard\s+meeting\b.*financial",
    ],
    EventCategory.DIVIDEND: [
        r"\bcash\s+dividend\b",
        r"\binterim\s+dividend\b",
        r"\bfinal\s+dividend\b",
        r"\bdividend\s+payout\b",
        r"\bdividend\s+(?:announced|declared|approved|per\s+share)\b",
        r"\bbook\s+closure\b.*dividend",
        r"\bpayout\s+of\s+pkr\b",
    ],
    EventCategory.BONUS_ISSUE: [
        r"\bbonus\s+shares?\b",
        r"\bbonus\s+issue\b",
        r"\bright\s+shares?\b",
        r"\brights?\s+issue\b",
        r"\bright\s+entitlements?\b",
    ],
    EventCategory.DISCOVERY: [
        r"\b(?:hydrocarbon|gas|oil)\s+discover(?:y|ies)\b",
        r"\bdiscover(?:y|ed)\s+of\s+(?:oil|gas|hydrocarbons?)\b",
        r"\bwell\s+test\b",
        r"\bexploration\s+well\b",
        r"\bflow\s+rate\b",
        r"\bmmscfd\b",
        r"\bbopd\b",
        r"\breserves?\s+discovered\b",
    ],
    EventCategory.PRODUCTION_CHANGE: [
        r"\bplant\s+shutdown\b",
        r"\bmaintenance\s+(?:shutdown|turnaround)\b",
        r"\bproduction\s+(?:halt|suspension|halted|stopped)\b",
        r"\bcapacity\s+expansion\b",
        r"\bcommercial\s+operations?\b",
        r"\bcommercial\s+production\b",
        r"\bsuspension\s+of\s+operations?\b",
        r"\bclosure\s+of\s+plant\b",
    ],
    EventCategory.REGULATORY: [
        r"\bsecp\s+(?:notice|directive|order|inquiry|approval)\b",
        r"\bpsx\s+circular\b",
        r"\bregulatory\s+approval\b",
        r"\bcourt\s+stay\b",
        r"\bstay\s+order\b",
        r"\bnepra\s+tariff\b",
        r"\bsbp\s+directive\b",
        r"\bcompetition\s+commission\b",
        r"\bshow[- ]cause\s+notice\b",
        r"\bpenalty\s+imposed\b",
        r"\bdelisting\b",
    ],
    EventCategory.MACRO_INTEREST_RATE: [
        r"\bmonetary\s+policy\b",
        r"\bpolicy\s+rate\b",
        r"\binterest\s+rates?\b",
        r"\bsbp\s+(?:keeps|hikes|cuts|reduces|raises)\s+(?:policy\s+)?rate\b",
        r"\binflation\s+(?:rate|surges|drops|eases)\b",
        r"\bcpi\s+inflation\b",
        r"\bimf\s+(?:tranche|review|mission|program|loan)\b",
        r"\bforeign\s+exchange\s+reserves\b",
        r"\bcurrent\s+account\s+(?:deficit|surplus)\b",
    ],
}


class EventClassifier:
    """Classifies unstructured text into financial and exchange event categories."""

    def __init__(self) -> None:
        self._compiled: dict[EventCategory, list[re.Pattern[str]]] = {}
        for cat, patterns in CATEGORY_PATTERNS.items():
            self._compiled[cat] = [re.compile(pat, re.IGNORECASE) for pat in patterns]

    def classify(self, text: str) -> tuple[str, float]:
        """Classify input text into the most probable event category with confidence.

        Args:
            text: Headline, title, or announcement text.

        Returns:
            Tuple of (category_name, confidence_score).
        """
        if not text or not text.strip():
            return EventCategory.OTHER.value, 0.0

        scores: dict[EventCategory, int] = {cat: 0 for cat in EventCategory}

        for cat, patterns in self._compiled.items():
            for pat in patterns:
                if pat.search(text):
                    scores[cat] += 1

        # Priority order when scores are tied: DISCOVERY/DIVIDEND/BONUS/EARNINGS before general
        top_cat = EventCategory.OTHER
        max_score = 0

        # Check in priority order
        priority_order = [
            EventCategory.DISCOVERY,
            EventCategory.BONUS_ISSUE,
            EventCategory.DIVIDEND,
            EventCategory.EARNINGS,
            EventCategory.PRODUCTION_CHANGE,
            EventCategory.REGULATORY,
            EventCategory.MACRO_INTEREST_RATE,
        ]

        for cat in priority_order:
            if scores[cat] > max_score:
                max_score = scores[cat]
                top_cat = cat

        if max_score == 0:
            return EventCategory.OTHER.value, 0.5

        # Calculate confidence: higher when distinct matches exist
        confidence = min(0.6 + (max_score * 0.15), 0.95)
        return top_cat.value, round(confidence, 2)
