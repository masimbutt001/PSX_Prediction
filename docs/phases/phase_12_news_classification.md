# Phase 12 — News Classification & NLP

## 1. Objective
Extract structured quantitative signals from unstructured financial text. Map news and announcements to specific PSX stock symbols, compute sentiment scores, and classify event types (e.g., earnings, dividends, discoveries).

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/news/
├── classifier.py                      # Event category classifier
├── sentiment.py                       # Lexicon and lightweight sentiment analyzer
└── entity_matcher.py                  # Symbol & company alias matching engine

tests/unit/
└── test_news_nlp.py                   # Matching accuracy and sentiment test cases
```

---

## 3. Detailed Specifications

### 3.1 Entity Matching (`entity_matcher.py`)
- Maps company names and aliases to PSX symbols:
  - `"Oil & Gas Development Company"`, `"OGDC"` $\to$ `OGDC`
  - `"Fauji Fertilizer"`, `"FFC"` $\to$ `FFC`
  - `"United Bank"`, `"UBL"` $\to$ `UBL`

### 3.2 Event Classification (`classifier.py`)
- Categorizes articles into classes:
  - `EARNINGS`, `DIVIDEND`, `BONUS_ISSUE`, `DISCOVERY`, `PRODUCTION_CHANGE`, `REGULATORY`, `MACRO_INTEREST_RATE`, `OTHER`.

### 3.3 Sentiment Scoring (`sentiment.py`)
- Financial lexicon analyzer (e.g., Loughran-McDonald financial dictionary or FinBERT).
- Outputs normalized sentiment score between $-1.0$ (strongly negative) and $+1.0$ (strongly positive).

---

## 4. Testing Plan
- `test_alias_to_symbol_matching()`: Asserts variations of company names map to expected tickers.
- `test_sentiment_polarity_sanity()`: Verifies positive phrases ("record profit surge", "dividend increase") score $> 0$ and negative ("sharp loss", "plant shutdown") score $< 0$.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_news_nlp.py -v
psx news process
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Entities accurately matched to PSX symbols.
- [ ] Structured sentiments and event tags generated in `data/processed/news/`.
- [ ] Phase 12 completion report documented.
