# Phase 13 — News & Market Session Alignment (PSX Calendar)

## 1. Objective
Map each news article to the earliest PSX trading session in which its information could legitimately have been acted upon (`usable_from`). Incorporate the PSX market calendar, including Friday split sessions and holidays, with zero look-ahead bias.

---

## 2. Deliverables & Files to Create
```text
src/psx_predictor/news/
├── calendar.py                        # PSX market session calendar (trading hours, holidays, Friday split)
└── session_aligner.py                 # Maps publication timestamp to usable_from session

tests/unit/
└── test_news_alignment.py             # Calendar alignment and session boundary tests
```

---

## 3. Detailed Specifications

### 3.1 PSX Trading Schedule Rules
- **Mon – Thu Regular Session:** 09:15 to 15:30 PKT.
  - News before 09:15 $\to$ usable for today's session.
  - News between 09:15 and 15:30 $\to$ usable starting today's close or next session open.
  - News after 15:30 $\to$ usable for next trading day.
- **Friday Split Schedule:**
  - Session 1: 09:15 to 12:00 PKT.
  - Prayer Break: 12:00 to 14:30 PKT. News during this break becomes usable for **Friday Session 2**.
  - Session 2: 14:30 to 16:30 PKT.
  - News after 16:30 $\to$ usable for Monday.

### 3.2 Feature Aggregation
- Generates point-in-time features:
  - `sentiment_24h`, `sentiment_72h`
  - `news_count_24h`, `positive_count_24h`, `negative_count_24h`
  - `has_earnings_announcement_today`

---

## 4. Testing Plan
- `test_after_hours_news_aligned_to_next_day()`: Article at 17:00 on Tuesday must have `session_target_date` as Wednesday.
- `test_friday_prayer_break_aligned_to_session_2()`: Article at 13:00 on Friday must align to Friday Session 2.
- `test_weekend_news_aligned_to_monday()`: Article at 14:00 on Sunday must align to Monday.

---

## 5. Validation Commands
```bash
pytest tests/unit/test_news_alignment.py -v
psx news align
ruff check .
```

---

## 6. Phase Completion Exit Criteria
- [ ] Friday prayer break and after-hours alignment verified.
- [ ] Point-in-time news features generated with zero future leakage.
- [ ] Phase 13 completion report documented.
