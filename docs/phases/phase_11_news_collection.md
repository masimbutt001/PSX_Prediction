# Phase 11 — News & Company Announcements Collection

## 1. Objective
Build decoupled data collectors for official PSX corporate announcements and reputable Pakistani financial news RSS feeds. Persist raw payloads in immutable JSON/Parquet storage before any natural language processing.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/news/
├── __init__.py
├── rss_collector.py                   # Financial news RSS feed collector (Business Recorder, Dawn, etc.)
└── dps_announcements.py               # PSX DPS official announcements collector

tests/unit/
└── test_news_collection.py            # Deduplication, parsing, and mocked network tests
```

---

## 3. Detailed Specifications

### 3.1 Data Sources
1. **Financial News Feeds:**
   - RSS feeds from Pakistani financial outlets (e.g., Business Recorder, Profit by Pakistan Today, Dawn Financial).
2. **PSX Company Announcements:**
   - Scrapes or fetches from PSX Data Portal (`dps.psx.com.pk/announcements`).
   - Extracts: Headline, company code, announcement category, timestamp, document link.

### 3.2 Raw Persistence & Deduplication
- Path: `data/raw/news/` and `data/raw/announcements/`.
- Deduplication key: MD5/SHA256 content hash of `(source, headline, published_at)`.
- Re-fetching the same feeds produces zero duplicate raw files.

---

## 4. Testing Plan
- `test_rss_parser_extracts_fields()`: Feeds sample XML fixture and checks headline, date, and link extraction.
- `test_news_deduplication()`: Verifies duplicate articles are detected and ignored.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_news_collection.py -v
psx news fetch --dry-run
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] News and announcement collectors functioning with mocks.
- [ ] Raw deduplicated records stored safely.
- [ ] Phase 11 completion report documented.
