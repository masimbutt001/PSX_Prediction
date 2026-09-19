# Walkthrough — Phase 15: Combined Multi-Modal Feature Set & Ablations

**Status:** Completed & Validated  
**Date:** September 2026  
**Specification:** [phase_15_combined_features.md](../phase_15_combined_features.md)

---

## 1. Objective Accomplished
Engineered a unified multi-modal point-in-time analytical feature store and an empirical ablation study testing framework:
1. **Multi-Modal Feature Merger (`MultiModalFeatureMerger`)**:
   - Synthesizes all three foundational information layers for each PSX security:
     $$\text{FeatureVector}(S, t) = \left[ \text{Tech}(S, t), \text{News}(S, t), \text{Macro}(t), \text{Interactions}(S, t) \right]$$
   - **Technical Indicators**: Vectorized OHLCV indicators (RSI, MACD, Bollinger Bands, ATR, SMAs, EMAs, volume dynamics) + supervised classification/return targets.
   - **Market-Aligned News**: Point-in-time 24h/72h sentiment polarity, article counts, directional signal volumes, and binary flags for high-impact company disclosures (`has_earnings_announcement_today`, `has_dividend_announcement_today`, `has_discovery_announcement_today`).
   - **Macroeconomic Drivers**: SBP Policy Rate, 6M KIBOR, PBS YoY CPI Inflation, USD/PKR spot rate, Brent Crude Oil, and KSE-100 benchmark returns joined via retrospective as-of backward matching with zero look-ahead bias.
   - **Cross-Modal Interaction & Relative Alpha Features**:
     - `rel_kse100_ret_1d`: Idiosyncratic asset excess return over the market benchmark ($\Delta \ln P_S - \Delta \ln I_{\text{KSE}}$).
     - `policy_rate_spread`: Banking liquidity risk premium ($\text{KIBOR} - \text{SBP}$).
     - `real_interest_rate`: Real economic cost of capital ($\text{SBP} - \text{CPI}$).
2. **Automated Ablation Study Runner (`AblationStudyRunner`)**:
   - Implements 5 standardized experimental configurations evaluated on **strictly identical chronological splits**:
     - **EXP-1**: `Price Only` (Baseline technical indicators on OHLCV).
     - **EXP-2**: `Price + Relative Market` (Technical + KSE-100 benchmark & relative return).
     - **EXP-3**: `Price + News` (Technical + news polarity & corporate announcements).
     - **EXP-4**: `Price + Macro` (Technical + SBP policy rates, KIBOR, CPI, FX, Oil, & spreads).
     - **EXP-5**: `Price + News + Macro (Full)` (Holistic multi-modal feature set).
   - Computes machine learning metrics (Accuracy, F1-macro, ROC-AUC, Brier score) and trading simulation metrics (Total Return, Annualized Sharpe Ratio, Max Drawdown, Win Rate).
   - Quantifies the empirical delta ($\Delta \text{Acc}$, $\Delta \text{Sharpe}$) from adding News and Macro signals over Price alone.
3. **Storage & CLI Commands**:
   - Atomic serialization to `data/features/combined/{SYMBOL}_combined.parquet`.
   - CLI commands: `psx features merge [--symbol/--symbols/--all]` and `psx model ablation [--symbol SYMBOL] [--model logistic|rf|xgb]`.

---

## 2. Deliverables & Files Created / Modified

| File | Purpose |
| :--- | :--- |
| [`src/psx_predictor/features/merger.py`](../../../src/psx_predictor/features/merger.py) | `MultiModalFeatureMerger` combining technical, news, macro, and interaction layers with zero lookahead. |
| [`src/psx_predictor/features/ablations.py`](../../../src/psx_predictor/features/ablations.py) | `AblationStudyRunner`, `AblationConfig`, and `AblationExperimentResult` running 5-configuration empirical tests. |
| [`src/psx_predictor/features/__init__.py`](../../../src/psx_predictor/features/__init__.py) | Exported merger and ablation public APIs. |
| [`src/psx_predictor/cli/main.py`](../../../src/psx_predictor/cli/main.py) | Registered `psx features merge` and `psx model ablation` CLI commands. |
| [`tests/unit/test_merger.py`](../../../tests/unit/test_merger.py) | 6 comprehensive unit tests verifying retrospective joins, missing news handling, configuration schemas, runner execution, and CLI commands. |

---

## 3. Architecture & Modality Merging Flow

```mermaid
graph TD
    subgraph Modalities
        T[Technical Features + Targets<br/>data/features/technical/] --> M[MultiModalFeatureMerger]
        N[News Sentiment & Disclosures<br/>data/features/news/] --> M
        MC[Macro Daily Drivers<br/>data/processed/macro/] --> M
    end

    subgraph Feature Join Engine
        M --> AS_OF[pd.merge_asof backward on date]
        AS_OF --> NEUTRAL[Zero-fill missing news]
        NEUTRAL --> INTER[Compute Interactions: rel_kse100, spreads]
        INTER --> COMBINED[(data/features/combined/{SYMBOL}_combined.parquet)]
    end

    subgraph Ablation Framework
        COMBINED --> ABL[AblationStudyRunner]
        ABL --> E1[EXP-1: Price Only]
        ABL --> E2[EXP-2: Price + Relative Market]
        ABL --> E3[EXP-3: Price + News]
        ABL --> E4[EXP-4: Price + Macro]
        ABL --> E5[EXP-5: Price + News + Macro]
        E1 & E2 & E3 & E4 & E5 --> COMP[Empirical Comparison Table + Deltas]
    end
```

---

## 4. Empirical Validation & Results

### 4.1 Automated Test Suite
All 134 tests across the complete test suite pass cleanly:
```bash
$ uv run pytest -v
========================== 134 passed, 2 deselected in 14.33s ==========================
```

Unit tests (`tests/unit/test_merger.py`):
- `test_asof_join_does_not_peek_future`: Confirms that a January CPI release published on February 1st is completely invisible to January 15th trading sessions.
- `test_merger_combines_all_modalities`: Validates successful union of technical, news, macro, and interaction layers into combined parquet.
- `test_merger_handles_missing_news_gracefully`: Verifies neutral defaults (sentiment=0.0, count=0.0) when a symbol has no news.
- `test_ablation_configurations_load`: Validates all 5 configuration schemas filter correct column subsets.
- `test_ablation_study_runner_execution`: Confirms identical chronological train/test sample sizes and valid delta metrics.
- `test_cli_features_merge_and_ablation`: Verifies `psx features merge` and `psx model ablation` execute cleanly with 0 exit code.

### 4.2 Type Checking & Linting
```bash
$ uv run ruff check .
All checks passed!

$ uv run mypy src
Success: no issues found in 60 source files
```

### 4.3 Live CLI Verification

#### Multi-Modal Feature Merge:
```bash
$ uv run psx features merge --symbol OGDC
Starting multi-modal feature merge | Target(s): OGDC | Mode: PERSIST
[OGDC] Starting multi-modal point-in-time feature merge...
[OGDC] Atomically persisted 1240 combined rows (70 cols) to data/features/combined/OGDC_combined.parquet

                     Multi-Modal Feature Merge Summary                      
+--------------------------------------------------------------------------+
| Symbol | Status  | Sessions | Feats (T/N/M/I) |        Date Span         |
|--------+---------+----------+-----------------+--------------------------|
| OGDC   | SUCCESS |     1240 |    49/9/9/3     | 2021-09-05 -> 2026-09-03 |
+--------------------------------------------------------------------------+
```

#### Empirical Ablation Study (Logistic Regression):
```bash
$ uv run psx model ablation --symbol OGDC --model logistic
Empirical Multi-Modal Ablation Study: OGDC (LOGISTIC on target_next_day_dir)
+-----------------------------------------------------------------------------+
| Configuration        | Feats |   Acc | +/-Acc |    F1 |   AUC | Return | Sharpe | +/-Sh | Win% |
|----------------------+-------+-------+--------+-------+-------+--------+--------+-------+------|
| Price Only           |    20 | 52.4% |     -- | 0.524 | 0.508 |  25.1% |   0.62 |    -- | 53.6%|
| Price + Relative     |    23 | 53.8% |  +1.4% | 0.538 | 0.537 |  26.0% |   0.70 | +0.08 | 54.5%|
| Market               |       |       |        |       |       |        |        |       |      |
| Price + News         |    29 | 52.4% |  +0.0% | 0.524 | 0.508 |  25.1% |   0.62 | +0.00 | 53.6%|
| Price + Macro        |    32 | 50.0% |  -2.4% | 0.397 | 0.512 |  20.5% |   0.43 | -0.19 | 50.0%|
| Price + News + Macro |    41 | 50.0% |  -2.4% | 0.397 | 0.512 |  20.5% |   0.43 | -0.19 | 50.0%|
| (Full)               |       |       |        |       |       |        |        |       |      |
+-----------------------------------------------------------------------------+
```

#### Empirical Ablation Study (Random Forest):
```bash
$ uv run psx model ablation --symbol OGDC --model rf
Empirical Multi-Modal Ablation Study: OGDC (RF on target_next_day_dir)
- Price + News improved accuracy over Price Only by +0.5% (50.5% -> 51.0%).
- Price + News improved macro F1 score from 0.360 to 0.390 (+0.030).
- Price + News increased total backtest return from +2.3% to +3.9% (+1.6%).
```
