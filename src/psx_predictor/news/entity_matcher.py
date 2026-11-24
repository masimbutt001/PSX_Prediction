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
    "ENGROH": [
        "Engro Holdings Limited",
        "Engro Holdings",
        "Engro Corporation Limited",
        "Engro Corporation",
        "Engro Corp",
        "Engro",
        "ENGRO",
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
    "AIRLINK": [
        "Air Link Communication Limited",
        "Air Link Communication",
        "Airlink Communication",
        "AIRLINK",
    ],
    "ATRL": [
        "Attock Refinery Limited",
        "Attock Refinery",
        "ATRL",
    ],
    "CPHL": [
        "Citi Pharma Limited",
        "Citi Pharma Ltd",
        "Citi Pharma",
        "CPHL",
    ],
    "DGKC": [
        "D.G. Khan Cement Company Limited",
        "D.G. Khan Cement",
        "DG Khan Cement",
        "DGKC",
    ],
    "EFERT": [
        "Engro Fertilizers Limited",
        "Engro Fertilizers",
        "Engro Fertilizer",
        "EFERT",
    ],
    "FCCL": [
        "Fauji Cement Company Limited",
        "Fauji Cement Company",
        "Fauji Cement",
        "FCCL",
    ],
    "FFL": [
        "Fauji Foods Limited",
        "Fauji Foods",
        "FFL",
    ],
    "GAL": [
        "Ghandhara Automobiles Limited",
        "Ghandhara Automobiles",
        "GAL",
    ],
    "GHNI": [
        "Ghandhara Industries Limited",
        "Ghandhara Industries",
        "GHNI",
    ],
    "HCAR": [
        "Honda Atlas Cars (Pakistan) Limited",
        "Honda Atlas Cars",
        "Honda Atlas",
        "HCAR",
    ],
    "MARI": [
        "Mari Energies Limited",
        "Mari Energies",
        "Mari Petroleum Company Limited",
        "Mari Petroleum",
        "MARI",
    ],
    "MEBL": [
        "Meezan Bank Limited",
        "Meezan Bank",
        "MEBL",
    ],
    "MLCF": [
        "Maple Leaf Cement Factory Limited",
        "Maple Leaf Cement Factory",
        "Maple Leaf Cement",
        "MLCF",
    ],
    "NML": [
        "Nishat Mills Limited",
        "Nishat Mills",
        "NML",
    ],
    "NRL": [
        "National Refinery Limited",
        "National Refinery",
        "NRL",
    ],
    "PAEL": [
        "Pak Elektron Limited",
        "Pak Elektron",
        "PEL",
        "PAEL",
    ],
    "PRL": [
        "Pakistan Refinery Limited",
        "Pakistan Refinery",
        "PRL",
    ],
    "PSO": [
        "Pakistan State Oil Company Limited",
        "Pakistan State Oil",
        "PSO",
    ],
    "SAZEW": [
        "Sazgar Engineering Works Limited",
        "Sazgar Engineering Works",
        "Sazgar Engineering",
        "Sazgar",
        "SAZEW",
    ],
    "SEARL": [
        "The Searle Company Limited",
        "Searle Company Limited",
        "Searle Company",
        "Searle",
        "SEARL",
    ],
    "SNGP": [
        "Sui Northern Gas Pipelines Limited",
        "Sui Northern Gas Pipelines",
        "Sui Northern Gas",
        "SNGPL",
        "SNGP",
    ],
    "SSGC": [
        "Sui Southern Gas Company Limited",
        "Sui Southern Gas Company",
        "Sui Southern Gas",
        "SSGCL",
        "SSGC",
    ],
    "TREET": [
        "Treet Corporation Limited",
        "Treet Corporation",
        "TREET",
    ],
}

# Tickers that are also common lowercase English words or roots
CASE_SENSITIVE_TICKERS: set[str] = {
    "LUCK",
    "SYS",
    "ENGRO",
    "ENGROH",
    "AIR",
    "PACE",
    "MARI",
    "GAL",
    "TREET",
    "PEL",
}


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
