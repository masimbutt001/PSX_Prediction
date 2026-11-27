# PSX Stock Research Prompt (for Google Antigravity)

Copy everything between the two `====` lines into Antigravity, with the
`D:\Development\PSX_Prediction` folder open as the workspace. Change the
`SYMBOLS` line first. Everything else can stay as it is.

After it finishes, run `uv run psx research validate` to check the files, then open the
Research page in the dashboard (`uv run psx dashboard`). The app reads schema version 1.0
(`D:\Development\PSX_Prediction\src\psx_predictor\research\schemas.py`); if you change the schema here, change it there too.

====================================================================

## Task

Project root: `D:\Development\PSX_Prediction`. Every path in this prompt is absolute; use them exactly as
written, whatever folder the terminal happens to be in.

You are an equity research analyst covering the Pakistan Stock Exchange (PSX).
Research each stock listed below using live web sources, then save the results
as files in this workspace. The files are read by a Python app, so the JSON
must follow the schema below exactly.

SYMBOLS: OGDC
(Comma separated. Supported: the KMI-30 universe in D:\Development\PSX_Prediction\config\stocks.yaml, e.g. AIRLINK, ATRL, CPHL, DGKC, EFERT, ENGROH, FCCL, FFC, FFL, GAL, GHNI, HCAR, HUBC, LUCK, MARI, MEBL, MLCF, NML, NRL, OGDC, PAEL, PPL, PRL, PSO, SAZEW, SEARL, SNGP, SSGC, SYS, TREET)

RESEARCH_DATE: today's date in Pakistan time (Asia/Karachi), format YYYY-MM-DD.

## Output files (one pair per symbol)

1. `D:\Development\PSX_Prediction\data\research\<SYMBOL>_<RESEARCH_DATE>.json` – machine readable, strict schema below.
2. `D:\Development\PSX_Prediction\data\research\<SYMBOL>_<RESEARCH_DATE>.md` – human readable report.

Example: `D:\Development\PSX_Prediction\data\research\OGDC_2026-11-23.json` and `D:\Development\PSX_Prediction\data\research\OGDC_2026-11-23.md`.
Create the `D:\Development\PSX_Prediction\data\research` folder if it does not exist. If files for the same
symbol and date already exist, overwrite them. Do not write anywhere else, do
not edit any source code, and do not run git commands.

## Research checklist (cover all of these)

1. **Company snapshot**: business lines, sector, market cap, free float, last close, 52 week range.
2. **Latest financials**: most recent quarterly and annual results (Pakistan fiscal year usually ends 30 June; banks and some others use December). Revenue, net profit, EPS, margins, growth YoY, balance sheet health.
3. **Earnings history**: last 4 to 8 reported periods, with announcement dates and whether results beat, matched or missed broker expectations.
4. **Dividends and payouts**: cash dividends, bonus shares and right shares for the last 3 years, with announcement and book closure dates; trailing 12 month dividend yield and payout ratio.
5. **Valuation**: P/E, P/B, EV/EBITDA, dividend yield, compared with sector peers on PSX and with the stock's own history. Give a fair value estimate only if you can justify the method.
6. **Sector and macro**: sector drivers (for example oil prices and circular debt for E&P, gas prices and urea pricing for fertilizer, coal and construction demand for cement, policy rate and ADR/tax for banks, capacity payments for power, IT exports and USD/PKR for tech, construction and gas supply for glass). Link to SBP policy rate, inflation, USD/PKR and IMF programme where relevant.
7. **News**: material news from the last 90 days, newest first, with date, short summary and a sentiment score.
8. **Corporate announcements**: PSX notices (board meetings, material information, financial results, book closures) from the last 90 days.
9. **Upcoming events**: known or expected dates in the next 90 days (board meetings, results, book closure, AGM, MPC policy decision, federal budget).
10. **Catalysts**: specific things that could move the price up or down, with direction, impact and rough probability.
11. **Risks**: regulatory, macro, FX, commodity, circular debt, governance, liquidity, legal, political, operational.
12. **Management and ownership**: CEO, chairman, major shareholders (government, sponsors, foreign), recent changes, governance issues.
13. **Analyst views**: recent broker ratings and target prices (for example Topline, Arif Habib, JS Global, AKD, Insight, Optimus, IGI) with dates.
14. **Market sentiment**: tone of news, broker notes and investor forums, and foreign or local institutional flows if reported.

### Preferred sources

- PSX Data Portal (dps.psx.com.pk) company page, announcements and payouts
- Company investor relations pages and published financial statements
- SECP, SBP, PBS, OGRA, NEPRA, Ministry of Finance
- Business Recorder, Dawn, The News, Express Tribune, Profit (Pakistan Today), Mettis Global, Arab News Pakistan, Reuters, Bloomberg
- Brokerage research summaries; data sites such as Sarmaaya, Investing.com, StockAnalysis, MarketScreener

## Rules (very important)

1. **No look-ahead.** Use only information published on or before RESEARCH_DATE. Record the date of the newest piece of information you used in `data_as_of`.
2. **Cite everything.** Every news item, number group, earnings entry, dividend, catalyst, risk and analyst view lists `source_ids` that point to entries in `sources`. Each source has a real URL and a published date (or `null` if unknown).
3. **Never invent numbers.** If you cannot find or verify a value, use `null` and add a line to `data_gaps` saying what is missing. A `null` is far better than a guess.
4. **Freshness.** Prefer the latest filings. If two sources disagree, use the official filing (PSX or company) and mention the conflict in `data_gaps`.
5. **JSON number format.** Plain JSON numbers only: no commas, no `%`, no `Rs`, no `PKR`, no text like "approx". Percentages are written as percent numbers (12.5 means 12.5%). Money fields say their unit in the field name (`_pkr` = rupees, `_pkr_bn` = billions of rupees, `_pkr_mn` = millions of rupees).
6. **Dates.** Always `YYYY-MM-DD`. If only a month is known use the first day of the month and note it in `data_gaps`.
7. **Enums.** Use exactly the uppercase values listed in the schema. Do not add new values.
8. **Same keys every time.** Every key in the schema must be present (including `video_brief`), even if its value is `null` or an empty list. Do not add extra keys at the top level (put anything extra in `notes`).
9. **Balanced view.** List at least 3 risks and at least 2 catalysts. Do not be promotional.
10. **This is research, not financial advice.** Keep the tone factual.

## Scoring rubric (fill `scores` consistently so scores can be compared over time)

Score on the evidence you found, not on general reputation.

| Field | Range | Meaning |
|---|---|---|
| `overall_sentiment` | -1.0 to 1.0 | Net tone of everything researched. -1 very negative, 0 neutral, +1 very positive |
| `news_sentiment` | -1.0 to 1.0 | Average of the last 30 days of news, weighted toward newer and more important items. 0 if no news |
| `outlook_short_term` | -1.0 to 1.0 | Expected direction over the next 1 to 4 weeks |
| `outlook_medium_term` | -1.0 to 1.0 | Expected direction over the next 3 to 12 months |
| `fundamental_score` | 0 to 100 | Overall business quality and earnings strength. 50 = sector average |
| `valuation_score` | 0 to 100 | Higher = cheaper / more attractive vs peers and history. 50 = fairly valued |
| `growth_score` | 0 to 100 | Revenue and earnings growth momentum. 50 = in line with inflation |
| `dividend_score` | 0 to 100 | Yield, consistency and safety of payouts. 0 = no dividend |
| `financial_health_score` | 0 to 100 | Balance sheet strength, liquidity, receivables (circular debt) and leverage |
| `management_governance_score` | 0 to 100 | Quality of management, disclosure and governance |
| `risk_score` | 0 to 100 | Higher = riskier. 50 = typical PSX large cap |
| `confidence` | 0.0 to 1.0 | How complete and recent your sources are. Below 0.5 if key data is missing or old |

Anchor points for -1 to 1 scores: -1.0 severe negative, -0.5 clearly negative, -0.2 slightly negative, 0 neutral or mixed, 0.2 slightly positive, 0.5 clearly positive, 1.0 exceptionally positive.

`stance` should agree with the scores: STRONG_BUY and BUY need a positive `outlook_medium_term`; SELL and STRONG_SELL need a negative one.

## JSON schema (version 1.0)

Write exactly this structure. Types are shown as placeholders. `number|null`
means a JSON number or `null`. Lists may be empty `[]` but must be present.

```json
{
  "schema_version": "1.0",
  "symbol": "OGDC",
  "company_name": "string",
  "sector": "string",
  "research_date": "YYYY-MM-DD",
  "data_as_of": "YYYY-MM-DD",
  "generated_by": "google-antigravity",
  "model_name": "string (the AI model you are running as, or null)",

  "price_context": {
    "last_close_pkr": "number|null",
    "last_close_date": "YYYY-MM-DD|null",
    "high_52w_pkr": "number|null",
    "low_52w_pkr": "number|null",
    "market_cap_pkr_bn": "number|null",
    "shares_outstanding_mn": "number|null",
    "free_float_pct": "number|null",
    "avg_daily_volume_shares": "number|null"
  },

  "scores": {
    "overall_sentiment": "number -1..1",
    "news_sentiment": "number -1..1",
    "outlook_short_term": "number -1..1",
    "outlook_medium_term": "number -1..1",
    "fundamental_score": "number 0..100",
    "valuation_score": "number 0..100",
    "growth_score": "number 0..100",
    "dividend_score": "number 0..100",
    "financial_health_score": "number 0..100",
    "management_governance_score": "number 0..100",
    "risk_score": "number 0..100",
    "confidence": "number 0..1"
  },

  "recommendation": {
    "stance": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "horizon": "SHORT|MEDIUM|LONG",
    "fair_value_pkr": "number|null",
    "fair_value_method": "string|null (e.g. DCF, P/E multiple, DDM, broker consensus)",
    "upside_pct": "number|null (fair value vs last close)",
    "rationale": "string, 2-4 sentences"
  },

  "fundamentals": {
    "fiscal_year_end_month": "number 1..12|null",
    "latest_period": "string|null (e.g. 1QFY27, FY26, 9M2026)",
    "latest_period_end": "YYYY-MM-DD|null",
    "revenue_pkr_bn": "number|null",
    "revenue_growth_yoy_pct": "number|null",
    "net_income_pkr_bn": "number|null",
    "net_income_growth_yoy_pct": "number|null",
    "eps_pkr": "number|null",
    "eps_ttm_pkr": "number|null",
    "eps_growth_yoy_pct": "number|null",
    "gross_margin_pct": "number|null",
    "net_margin_pct": "number|null",
    "roe_pct": "number|null",
    "pe_ratio": "number|null",
    "pb_ratio": "number|null",
    "ev_ebitda": "number|null",
    "book_value_per_share_pkr": "number|null",
    "debt_to_equity": "number|null",
    "current_ratio": "number|null",
    "source_ids": ["S1"]
  },

  "earnings": [
    {
      "period": "string (e.g. 1QFY27)",
      "period_end": "YYYY-MM-DD|null",
      "announcement_date": "YYYY-MM-DD|null",
      "revenue_pkr_bn": "number|null",
      "net_income_pkr_bn": "number|null",
      "eps_pkr": "number|null",
      "eps_yoy_pct": "number|null",
      "vs_expectation": "BEAT|INLINE|MISS|UNKNOWN",
      "source_ids": ["S2"]
    }
  ],

  "dividends": {
    "trailing_12m_dps_pkr": "number|null",
    "dividend_yield_pct": "number|null",
    "payout_ratio_pct": "number|null",
    "history": [
      {
        "period": "string",
        "type": "INTERIM|FINAL|SPECIAL|BONUS|RIGHT",
        "amount_pkr": "number|null (cash per share; null for bonus/right)",
        "bonus_or_right_pct": "number|null",
        "announcement_date": "YYYY-MM-DD|null",
        "book_closure_start": "YYYY-MM-DD|null",
        "source_ids": ["S3"]
      }
    ]
  },

  "upcoming_events": [
    {
      "date": "YYYY-MM-DD|null",
      "date_is_estimate": true,
      "event_type": "EARNINGS|BOARD_MEETING|BOOK_CLOSURE|EX_DIVIDEND|AGM|EGM|POLICY_RATE|BUDGET|INDEX_REBALANCE|OTHER",
      "description": "string",
      "expected_impact": "POSITIVE|NEGATIVE|NEUTRAL|UNCERTAIN",
      "source_ids": ["S4"]
    }
  ],

  "catalysts": [
    {
      "title": "string",
      "description": "string",
      "direction": "POSITIVE|NEGATIVE",
      "impact": "LOW|MEDIUM|HIGH",
      "probability": "number 0..1",
      "timeframe": "SHORT|MEDIUM|LONG",
      "source_ids": ["S5"]
    }
  ],

  "risks": [
    {
      "title": "string",
      "category": "REGULATORY|MACRO|FX|COMMODITY|CIRCULAR_DEBT|GOVERNANCE|LIQUIDITY|COMPETITION|POLITICAL|LEGAL|OPERATIONAL|OTHER",
      "severity": "LOW|MEDIUM|HIGH",
      "likelihood": "LOW|MEDIUM|HIGH",
      "description": "string",
      "source_ids": ["S6"]
    }
  ],

  "news": [
    {
      "date": "YYYY-MM-DD",
      "headline": "string",
      "summary": "string, 1-2 sentences",
      "category": "EARNINGS|DIVIDEND|CORPORATE_ACTION|REGULATORY|MACRO|SECTOR|MANAGEMENT|LEGAL|PROJECT|OTHER",
      "sentiment": "number -1..1",
      "importance": "LOW|MEDIUM|HIGH",
      "source_ids": ["S7"]
    }
  ],

  "sector_view": {
    "name": "string",
    "outlook": "number -1..1",
    "key_drivers": ["string"],
    "peers": [
      {
        "symbol": "string (PSX symbol)",
        "pe_ratio": "number|null",
        "pb_ratio": "number|null",
        "dividend_yield_pct": "number|null",
        "note": "string|null"
      }
    ],
    "source_ids": ["S8"]
  },

  "management": {
    "ceo": "string|null",
    "chairman": "string|null",
    "major_shareholders": [
      { "name": "string", "stake_pct": "number|null" }
    ],
    "recent_changes": ["string"],
    "governance_notes": "string|null",
    "source_ids": ["S9"]
  },

  "analyst_views": [
    {
      "broker": "string",
      "date": "YYYY-MM-DD|null",
      "rating": "BUY|OVERWEIGHT|HOLD|NEUTRAL|UNDERWEIGHT|SELL|UNKNOWN",
      "target_price_pkr": "number|null",
      "source_ids": ["S10"]
    }
  ],

  "market_sentiment": {
    "forum_sentiment": "number -1..1|null",
    "institutional_flows_note": "string|null",
    "notes": "string|null"
  },

  "key_takeaways": ["string (3 to 7 short bullet sentences)"],

  "video_brief": {
    "hook": "string: one spoken opening line, under 25 words, that makes a viewer want to keep watching",
    "story": ["string: 4 to 6 short spoken sentences in order: the situation, what changed, why it matters, what to watch next"],
    "tone": "UPBEAT|BALANCED|CAUTIOUS|NEGATIVE",
    "chart_highlights": [
      {
        "chart": "SCORES|PRICE_TREND|PRICE_RANGE|VALUATION_VS_PEERS|EARNINGS|DIVIDENDS|RISK_GRID|EVENTS",
        "highlight": "string: which bar, value or point to emphasise and the one-line reason"
      }
    ],
    "pronunciations": [
      { "term": "string (as written, e.g. OGDC, Rs, 1QFY27, KIBOR)", "say": "string (how to say it, e.g. O G D C, rupees, first quarter of fiscal twenty-seven, kai-bor)" }
    ],
    "closing_line": "string: one spoken sentence that wraps up, before the disclaimer"
  },
  "data_gaps": ["string"],
  "notes": "string|null",

  "sources": [
    {
      "id": "S1",
      "title": "string",
      "publisher": "string",
      "url": "https://...",
      "published_date": "YYYY-MM-DD|null",
      "accessed_date": "YYYY-MM-DD",
      "type": "PSX_FILING|COMPANY_REPORT|BROKER_REPORT|NEWS|REGULATOR|DATA_PROVIDER|OTHER"
    }
  ]
}
```

In the real file replace every placeholder string (like `"number|null"`) with
an actual value. `date_is_estimate` is `true` or `false`.

## Video brief (`video_brief`)

This section is read by the video prompt (`D:\Development\PSX_Prediction\docs\research\antigravity_video_prompt.md`) to make the
narrated video sound like a human analyst, not a robot reading a table. Write it last,
after all other sections, and only from what you found.

- Write the way a good analyst talks to a colleague: plain words, contractions ("it's",
  "they've"), short and varied sentences, one idea per sentence. No bullet-speak, no
  "Furthermore", no "In conclusion".
- Round numbers for speech ("about 6.6 times earnings", "roughly a 20 percent upside"),
  but never change their meaning, and every number must exist elsewhere in the JSON.
- `hook` should be specific to this stock today (a result, a payout, a risk, a valuation
  gap), not generic ("Let's look at OGDC").
- `chart_highlights`: 2 to 5 entries pointing at the most telling values, so the video can
  animate and emphasise them while the narrator talks about them.
- `pronunciations`: every ticker, abbreviation, fiscal period and local name a text-to-speech
  voice could get wrong (tickers letter by letter, "Rs" as "rupees", "bn" as "billion",
  Urdu or company names spelled phonetically).
- `story` must be a JSON array of separate sentence strings, not one paragraph string.
- `tone` must match the scores and stance. Keep it balanced: mention the main risk in `story`.

## Markdown report layout (`.md` file)

Use these headings in this order, and keep the numbers identical to the JSON:

```
# <SYMBOL> – <Company name> | Research <RESEARCH_DATE>
> Data as of <data_as_of>. Research only, not financial advice.

## Summary            (stance, fair value, upside, 3-7 key takeaways)
## Scores             (table of all scores with one line of reasoning each)
## Price & Valuation
## Financial Results  (table of recent periods)
## Dividends & Payouts
## Sector & Macro
## Recent News        (table: date, headline, sentiment, source)
## Upcoming Events
## Catalysts
## Risks
## Management & Ownership
## Analyst Views
## Data Gaps
## Sources            (numbered list: [S1] title, publisher, date, URL)
```

## Final self-check (do this before you finish)

1. Run this in the terminal from the workspace folder and fix any error it prints:

   ```
   python -c "import json,glob; [json.load(open(f, encoding='utf-8')) for f in glob.glob(r'D:\Development\PSX_Prediction\data\research\*.json')]; print('JSON OK')"
   ```

2. Confirm for each JSON file:
   - The file name matches `<SYMBOL>_<RESEARCH_DATE>.json` and the `symbol` and `research_date` inside match the file name.
   - Every key from the schema is present; no placeholder text is left.
   - All scores are inside their ranges; all enums use the allowed values.
   - Every `source_ids` entry exists in `sources`; every source has a URL.
   - No date is later than `research_date`, except in `upcoming_events`.
3. Reply with a short summary per symbol: stance, overall sentiment, risk score, confidence, and the paths of the files you saved.

====================================================================
