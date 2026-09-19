"""Entity matching engine mapping unstructured financial text to PSX stock symbols."""

import re
from typing import Optional

from psx_predictor.config.loader import load_config

DEFAULT_ALIASES: dict[str, list[str]] = {
    "OGDC": [
        "Oil & Gas Development Company Limited",
        "Oil and Gas Development Company Limited",
        "Oil & Gas Development Company",
        "Oil and Gas Development Company",
        "Oil & Gas Development",
        "Oil and Gas Development",
        "OGDCL",
        "OGDC",
    ],
    "PPL": [
        "Pakistan Petroleum Limited",
        "Pakistan Petroleum",
        "PPL",
    ],
    "ENGRO": [
        "Engro Corporation Limited",
        "Engro Corporation",
        "Engro Corp",
        "Engro",
    ],
    "FFC": [
        "Fauji Fertilizer Company Limited",
        "Fauji Fertilizer Company",
        "Fauji Fertilizer",
        "FFC",
    ],
    "LUCK": [
        "Lucky Cement Limited",
        "Lucky Cement",
        "LUCK",
    ],
    "HBL": [
        "Habib Bank Limited",
        "Habib Bank",
        "HBL",
    ],
    "MCB": [
        "MCB Bank Limited",
        "MCB Bank",
        "MCB",
    ],
    "UBL": [
        "United Bank Limited",
        "United Bank",
        "UBL",
    ],
    "HUBC": [
        "The Hub Power Company Limited",
        "The Hub Power Company",
        "Hub Power Company",
        "Hub Power",
        "Hubco",
        "HUBC",
    ],
    "SYS": [
        "Systems Limited",
        "Systems Ltd",
        "SYS",
    ],
    "GGL": [
        "Ghani Global Glass Ltd.",
        "Ghani Global Glass Limited",
        "Ghani Global Glass",
        "Ghani Global",
        "GGL",
    ],
}

# Tickers that are also common lowercase English words or roots
CASE_SENSITIVE_TICKERS: set[str] = {"LUCK", "SYS", "ENGRO", "AIR", "PACE"}


class EntityMatcher:
    """Matches company names, aliases, and ticker symbols within financial text."""

    def __init__(self, custom_aliases: Optional[dict[str, list[str]]] = None) -> None:
        self.aliases: dict[str, list[str]] = {}

        # 1. Ingest universe from config if available
        try:
            cfg = load_config()
            for s in cfg.stocks:
                sym = s.symbol.upper()
                self.aliases[sym] = [s.name, sym]
        except Exception:
            pass

        # 2. Ingest default comprehensive Pakistani corporate aliases
        for sym, alias_list in DEFAULT_ALIASES.items():
            if sym not in self.aliases:
                self.aliases[sym] = []
            for a in alias_list:
                if a not in self.aliases[sym]:
                    self.aliases[sym].append(a)

        # 3. Ingest custom overrides
        if custom_aliases:
            for sym, alias_list in custom_aliases.items():
                sym_up = sym.upper()
                if sym_up not in self.aliases:
                    self.aliases[sym_up] = []
                for a in alias_list:
                    if a not in self.aliases[sym_up]:
                        self.aliases[sym_up].append(a)

        # 4. Compile regex patterns
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for sym, alias_list in self.aliases.items():
            # Sort aliases longest first so greedy multi-word matches take precedence
            sorted_aliases = sorted(alias_list, key=len, reverse=True)
            for alias in sorted_aliases:
                clean_alias = alias.strip()
                if not clean_alias:
                    continue

                if clean_alias.upper() == sym and sym in CASE_SENSITIVE_TICKERS:
                    # Require exact case match for common English words like 'LUCK' or 'SYS'
                    pattern = re.compile(r"\b" + re.escape(clean_alias) + r"\b")
                else:
                    # Allow case-insensitive match for multi-word or unique corporate names
                    # Handle & and 'and' interchangeably
                    pat_str = re.escape(clean_alias).replace(r"\&", r"(?:&|and)")
                    pattern = re.compile(r"\b" + pat_str + r"\b", re.IGNORECASE)

                self._patterns.append((sym, pattern))

    def match(self, text: str) -> list[str]:
        """Extract all unique PSX symbols mentioned in the input text.

        Args:
            text: Headline, summary, or announcement body text.

        Returns:
            Alphabetically sorted list of matched ticker symbols.
        """
        if not text:
            return []

        matched: set[str] = set()
        for sym, pat in self._patterns:
            if pat.search(text):
                matched.add(sym)

        return sorted(matched)

    def match_first(self, text: str) -> Optional[str]:
        """Return the first matched symbol in the text, or None."""
        symbols = self.match(text)
        return symbols[0] if symbols else None
