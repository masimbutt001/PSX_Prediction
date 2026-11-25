"""Plain-language explanations behind the (?) buttons on every dashboard page.

Each topic says what the element is, what it tells you and how to read it. The text is
written from the code that fills the element, so keep it in step when that code changes.
`speech_text` turns a topic into the text the Listen button reads aloud.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class HelpTopic:
    title: str
    body: str  # Markdown: short paragraphs and "- " bullets, **bold** allowed


def _t(title: str, body: str) -> HelpTopic:
    lines = [ln.strip() for ln in body.strip().splitlines()]
    return HelpTopic(title=title, body="\n".join(lines))


HELP: dict[str, HelpTopic] = {
    # ------------------------------------------------------------------ header
    "hdr_freshness": _t(
        "Data freshness",
        """
        **What it is:** a status line that shows the newest trading day in your price files.

        **How to read it:**
        - A **green dot** and "all up to date" means every stock has prices for the newest
          session.
        - An **amber dot** lists the stocks that are behind. Their signals are based on older
          prices.
        - A **red dot** means there is no price data yet.

        **What to do:** use **Refresh** after a job finishes outside the dashboard, so the
        pages read the files again. To fetch missing days, use **Update prices** on the
        Pipeline page.
        """,
    ),
    # ---------------------------------------------------------------- overview
    "ov_kse100": _t(
        "KSE-100",
        """
        **What it is:** the main index of the Pakistan Stock Exchange, made of the 100
        biggest companies. It is read from the PSX website during the daily run.

        **What it tells you:** the big number is the index level at the last close. The line
        below is how much it moved that day.

        **How to read it:** green with an up arrow means the market as a whole rose, red with
        a down arrow means it fell. Use it as background: when the whole market falls, most
        single stocks fall with it.
        """,
    ),
    "ov_signals": _t(
        "Signals",
        """
        **What it is:** a count of today's model signals across all tracked stocks, in the
        order **BUY · HOLD · SELL**. For example "5 · 18 · 7" means 5 BUY, 18 HOLD and 7 SELL.

        **Where the signals come from:** for each stock the model estimates the chance that
        the next session closes **higher** than the last close.
        - **BUY** when that chance is **52% or more**.
        - **SELL** when it is **48% or less**.
        - **HOLD** when it is in between, which means the model has no clear view.

        **How to read it:** it gives the mood of the model for the next session. Many BUYs
        means it leans up on most stocks. A 52% chance is only slightly better than a coin
        flip, so treat a single signal as a weak hint, not a decision. Check the Performance
        page to see how often signals have been right.
        """,
    ),
    "ov_hit_rate": _t(
        "Hit rate",
        """
        **What it is:** the share of past predictions that turned out right, once the real
        close was known.

        **What counts as right:**
        - A **BUY** is right if the stock closed higher.
        - A **SELL** is right if it closed flat or lower.
        - A **HOLD** is right if the stock moved less than 0.75% either way.

        **How to read it:** 50% is what guessing would get. Above 50% is shown green, below
        is red. Compare it with the share of days that went up on the Performance page: if
        stocks rose on 55% of days, always saying BUY would already score about 55%. The
        number under it is how many predictions were scored; with few, the rate jumps around
        a lot.
        """,
    ),
    "ov_brier": _t(
        "Brier score",
        """
        **What it is:** a score for how good the model's percentages are, not just its
        direction. For each prediction it takes the gap between the stated chance of UP and
        what happened (1 if the stock rose, 0 if not), squares it, and averages them all.

        **How to read it:** **lower is better**. 0 would be perfect. Always saying 50%
        scores exactly **0.250**, which is a coin flip. So a score **below 0.250** (green)
        means the probabilities carry some real information, and **above 0.250** (red) means
        they do worse than saying 50% every time.
        """,
    ),
    "ov_usdpkr": _t(
        "USD / PKR",
        """
        **What it is:** how many Pakistani rupees one US dollar costs, from the daily macro
        update.

        **How to read it:** a rising number means the rupee got **weaker**. That is shown
        red, because a weaker rupee usually hurts importers and the market mood. A falling
        number means the rupee got stronger and is shown green. Exporters and companies with
        dollar income can gain from a weaker rupee.
        """,
    ),
    "ov_watchlist": _t(
        "Watchlist and signals",
        """
        **What it is:** one row for every stock you track, with its latest price, recent
        moves and the model's signal for the next session.

        **The columns:**
        - **Close (PKR):** last closing price.
        - **1D % and 5D %:** change over the last session and the last 5 sessions.
        - **30 sessions:** a tiny line of the last 30 closes, to see the trend at a glance.
        - **Signal:** BUY, HOLD or SELL for the next session.
        - **Chance of UP:** the model's probability that the next close is higher.
        - **Research:** the stance from your newest research file, if there is one.

        **How to use it:** filter by signal with the buttons on the right and by sector with
        the chips. Filters combine. Click a row to open that stock. **Export CSV** saves the
        full list as a spreadsheet file.
        """,
    ),
    "ov_sectors": _t(
        "Sector moves, last session",
        """
        **What it is:** the average one-day price change of the tracked stocks in each
        sector, for the last session.

        **How to read it:** green bars to the right rose, red bars to the left fell. The
        longest bars are the sectors that moved most. A sector average can come from only one
        or two stocks, so a big bar may just be one company's news. Hover a bar for the exact
        number.
        """,
    ),
    "ov_last_run": _t(
        "Last daily run",
        """
        **What it is:** the result of the most recent automatic daily update. It runs six
        steps in order: prices, news and announcements, macro data, features, predictions,
        and the audit that scores past predictions.

        **How to read it:** each step shows a green tick when it worked, a grey dash when it
        was skipped and a red mark when it failed. The line on top says when it ran and how
        long it took. If a step failed, the signals may be based on old data.

        **What to do:** **Run now** starts the full update in the background on this
        computer. **Open pipeline details** shows the error and lets you rerun one step.
        """,
    ),
    "ov_news": _t(
        "Latest news",
        """
        **What it is:** the five newest headlines collected from Business Recorder, Dawn and
        PSX company announcements.

        **How to read it:** each headline has a tag. **POS** means its words read as good
        news, **NEG** as bad news and **NEU** as neutral. The tag comes from a word list, not
        from reading the article, so it can miss sarcasm or context. Under each headline you
        see the source, the stock it was matched to (or "Market") and the date.
        """,
    ),
    # ------------------------------------------------------------------ stocks
    "sd_title": _t(
        "Price and actions",
        """
        **What it is:** the stock's name, sector and last closing price, with the change from
        the session before.

        **The buttons:**
        - **Refresh prediction:** retrains the model on this stock's history now and logs a
          new prediction for the next session.
        - **Prices CSV:** saves all price history for this stock as a spreadsheet file.
        - **Update prices:** fetches the newest sessions for this stock in the background.
        """,
    ),
    "sd_chart": _t(
        "Price chart",
        """
        **What it is:** a candlestick chart of the daily prices. Each candle is one session:
        the box runs from the open to the close, and the thin line shows the high and low.
        A green candle closed higher than it opened, a red one closed lower.

        **The options:**
        - **1M to All:** how much history to show.
        - **SMA 20 / SMA 50:** the average close of the last 20 or 50 sessions. Price above a
          rising average suggests an uptrend.
        - **Bollinger:** a band two standard deviations around SMA 20. Price near the top band
          is stretched up, near the bottom band stretched down.
        - **Volume:** shares traded each day, in the lower panel.

        **The marker after the last candle** is the next-session forecast: a triangle up or
        down with the chance of UP.

        Drag to zoom into a period, double-click to zoom back out.
        """,
    ),
    "sd_stats": _t(
        "Key numbers",
        """
        - **52-week high and low:** the highest and lowest price over the last 252 sessions,
          about one year.
        - **Daily volatility, 20d:** how much the price typically moves in a day, measured
          over the last 20 sessions. 2% means a normal day moves about 2% up or down.
        - **Avg volume, 20d:** the average number of shares traded per day, in millions.
        - **1-year change:** the price change over the last 252 sessions.
        - **Sessions on file:** how many trading days of history are stored. More history
          gives the model more to learn from.
        """,
    ),
    "sd_next_session": _t(
        "Next session prediction",
        """
        **What it is:** the model's latest forecast for the next trading session, and the
        signal it gives.

        **How to read it:**
        - The **ring** shows the chance the next close is higher than the last one. 50% means
          no idea. The ring turns green at 55% or more and red at 45% or less.
        - The **chip** is the signal: BUY at 52% or more, SELL at 48% or less, HOLD in between.
        - **Confidence** is how far the chance is from 50%, scaled to 0 to 100%. A 60% chance
          gives 20% confidence, a 50% chance gives 0%.
        - **Model** is the method used, normally XGBoost, retrained on this stock's own price
          history each time.
        - **Research** is the stance from your newest research file, if any.
        - **Hit rate, 60 sessions** is how many of this stock's last 60 scored predictions
          were right.

        If research disagrees with the model's direction, a warning appears below.
        """,
    ),
    "sd_drivers": _t(
        "Why the model thinks this",
        """
        **What it is:** the five inputs the model leaned on most, such as RSI, moving
        averages or volume measures.

        **How to read it:** a longer bar means the model used that input more when it learned
        from this stock's history. It shows **which** inputs matter, not whether each one
        pushes the price up or down today. Treat it as a guide to what the model is looking at.
        """,
    ),
    "sd_history": _t(
        "Prediction history",
        """
        **What it is:** the last 40 predictions for this stock, newest first.

        **The columns:** the **session** predicted, the **signal**, the **chance of UP** the
        model gave, the **actual move** of the close on that day, and the **result**.

        **Result:** **Correct** in green, **Missed** in red, or **Pending** when that
        session's close is not in the data yet. A BUY is correct if the stock rose, a SELL if
        it was flat or fell, and a HOLD if it moved less than 0.75%.
        """,
    ),
    "sd_news": _t(
        "News for this stock",
        """
        **What it is:** the 15 newest headlines and announcements matched to this stock.

        **How to read it:** the **POS / NEG / NEU** tag and the sentiment number (from -1, very
        negative, to +1, very positive) come from a word list applied to the headline. They are
        a quick hint only. Read the headline itself before acting on it.
        """,
    ),
    "sd_research": _t(
        "Research used in the prediction",
        """
        **What it is:** the research file that was applied to this stock's latest prediction.

        - **Stance:** the research recommendation, such as buy, hold or sell.
        - **Outlook 1-4 weeks:** the short-term view from -1 (very negative) to +1 (very
          positive).
        - **Risk:** 0 to 100, higher is riskier.
        - **Weight today:** how much the research still counts. It halves every 30 days, is
          multiplied by the research confidence and drops to zero after 120 days.
        - **Research date:** when the research was written.
        """,
    ),
    # ------------------------------------------------------------- performance
    "pf_window": _t(
        "Time window",
        """
        **What it is:** picks which scored predictions the numbers below use: those for
        sessions in the last **30 days**, the last **90 days**, or **all time**.

        **How to use it:** a short window shows how the model is doing lately. A long window
        has more predictions, so its numbers are more reliable and jump around less.
        """,
    ),
    "pf_hit_rate": _t(
        "Hit rate",
        """
        **What it is:** the share of scored predictions that were right in the chosen window.
        A BUY is right if the stock rose, a SELL if it was flat or fell, and a HOLD if it moved
        less than 0.75%.

        **How to read it:** green at 50% or more, red below. The fair comparison is the
        **Up days** card: if stocks rose on 55% of days, saying BUY every day would already
        score about 55%. A model is only useful if it beats that.
        """,
    ),
    "pf_brier": _t(
        "Brier score",
        """
        **What it is:** a grade for the model's percentages. For each prediction it squares
        the gap between the chance of UP it gave and what happened (1 for up, 0 for not up),
        then averages.

        **How to read it:** **lower is better**, 0 is perfect. Saying 50% every time scores
        exactly **0.250**. Below 0.250 (green) means the percentages contain some real
        information. Above 0.250 (red) means they are worse than a coin flip.
        """,
    ),
    "pf_ece": _t(
        "Calibration error",
        """
        **What it is:** how far the model's stated chances are from what really happened, on
        average. It sorts predictions into ten groups by their chance of UP (0-10%, 10-20% and
        so on), compares each group's average stated chance with the share that really went
        up, and averages the gaps, weighting bigger groups more.

        **How to read it:** 0.050 means the stated chances are off by about 5 percentage
        points. Under **0.05** is good (green), **0.05 to 0.10** is fair (amber), above
        **0.10** is poor (red). A low error means you can take a "60%" roughly at face value.
        """,
    ),
    "pf_up_days": _t(
        "Up days",
        """
        **What it is:** the share of scored sessions where the stock closed higher than the
        day before.

        **Why it matters:** it is the baseline to beat. If 55% of days were up, a lazy rule of
        always saying BUY would be right about 55% of the time. Compare it with the hit rate:
        a hit rate below this number means the model adds nothing over that lazy rule.
        """,
    ),
    "pf_calibration": _t(
        "Are the probabilities honest?",
        """
        **What it is:** a calibration chart. Predictions are grouped by the chance of UP the
        model gave. For each group, a dot shows the stated chance (across) against the share
        that really went up (up the side). Bigger dots have more predictions behind them.

        **How to read it:** the **dotted diagonal** is perfect honesty: when the model says
        60%, stocks rise 60% of the time.
        - Dots **above** the line: things went up more often than the model said, so it was
          too cautious.
        - Dots **below** the line: it was too optimistic.
        - A line of dots that is **flat** near the middle means the percentages tell you
          little.

        Hover a dot for its numbers. Small dots with few predictions can sit far from the
        line just by chance.
        """,
    ),
    "pf_track_record": _t(
        "Track record by stock",
        """
        **What it is:** the same scores, split by stock, for the chosen window, sorted with
        the best hit rate first.

        **The columns:** **Scored** is how many predictions have a known result, **Hit rate**
        is the share that were right, **Brier** is the probability score (below 0.250 beats a
        coin flip) and **Last scored** is the newest session scored.

        **How to read it:** stocks with only a few scored predictions can show very high or
        very low rates by luck. Trust rows with more predictions. **Export** saves every scored
        prediction as a spreadsheet file.
        """,
    ),
    "bt_settings": _t(
        "Backtest settings",
        """
        **What it is:** a backtest replays a trading rule on this stock's past prices, as if
        you had followed it, after PSX costs (brokerage, fees and slippage, about 0.27% each
        time you buy or sell).

        **The settings:**
        - **Stock:** which price history to use.
        - **Strategy:** **Swing pullback** buys dips inside an uptrend and holds for days to
          weeks. **SMA crossover 10/30** holds while the 10-day average is above the 30-day
          average. **1-day momentum** holds the day after any up day. **Buy and hold** buys on
          day one and never sells.
        - **Starting cash:** the amount the replay begins with.
        - **Risk-free rate:** what a safe deposit pays per year. It is only used for the
          Sharpe ratio.

        Past results do not guarantee future ones.
        """,
    ),
    "bt_swing_rules": _t(
        "Swing rules",
        """
        **What it is:** the knobs of the swing pullback strategy. The defaults follow the
        written rules.

        - **ADX minimum:** how strong the trend must be. Higher means fewer, stronger trends.
        - **RSI pullback zone:** how far RSI must dip to count as a pullback.
        - **Relative volume:** how much busier than normal the buy day must be. 1.2 means 20%
          above the 20-day average.
        - **Risk per trade:** the share of your money you can lose if the stop is hit. This
          sets how many shares are bought.
        - **Stop (x ATR):** how far below the entry the stop loss sits, in units of the
          average daily range.
        - **Take half at (R):** sell half once the gain is this many times the risk.
        - **Trail the rest with:** the line that follows the price up for the other half.
        - **Time stop:** sell after this many sessions; 0 turns it off.
        - **KSE-100 above SMA 50:** only buy when the whole market is in an uptrend.
        """,
    ),
    "bt_total_return": _t(
        "Total return",
        """
        **What it is:** how much the starting cash grew or shrank over the whole period, after
        all trading costs. +25% means 1,000,000 PKR became 1,250,000 PKR.

        **How to read it:** compare it with **buy and hold**, which simply buys on the first
        day and keeps the shares. A strategy that makes less than buy and hold needs another
        reason to use it, such as much smaller drawdowns.
        """,
    ),
    "bt_cagr": _t(
        "CAGR",
        """
        **What it is:** the compound annual growth rate: the steady yearly return that would
        turn the starting cash into the final value over the same number of years.

        **How to read it:** it lets you compare periods of different length. Compare it with
        a bank deposit or the risk-free rate you set.
        """,
    ),
    "bt_sharpe": _t(
        "Sharpe ratio",
        """
        **What it is:** return per unit of risk. It takes the daily returns above the
        risk-free rate, divides their average by how much they swing, and scales that to a
        year.

        **How to read it:** below 0 means it earned less than the safe rate. About 0.5 is
        modest, 1 is good and above 2 is rare. A higher Sharpe means a smoother ride for the
        same return.
        """,
    ),
    "bt_max_drawdown": _t(
        "Max drawdown",
        """
        **What it is:** the worst fall from a peak in portfolio value to the lowest point
        after it, before a new peak. -20% means at one point you were 20% below your best
        value so far.

        **How to read it:** it shows the pain you would have had to sit through. Smaller
        (closer to 0) is better. Compare it with buy and hold's drawdown.
        """,
    ),
    "bt_win_rate": _t(
        "Win rate",
        """
        **What it is:** the share of completed trades (buy, then sell) that made money after
        costs. The note under it shows how many trades closed.

        **How to read it:** a win rate under 50% can still make money if winners are bigger
        than losers, and a high win rate can still lose money if the few losses are large.
        Look at it together with the average trade and total return. With only a handful of
        trades it means little.
        """,
    ),
    "bt_avg_r": _t(
        "Average trade (R)",
        """
        **What it is:** the average profit per trade, measured in **R**, the amount risked on
        that trade (the distance from the entry to the stop loss).

        **How to read it:** +1R means the average trade made as much as it risked. +0.5R
        means half of the risk on average. Above 0 means the rules made money per trade over
        this period. Below 0 means they lost.
        """,
    ),
    "bt_exposure": _t(
        "Time in market",
        """
        **What it is:** the share of sessions where the strategy held shares.

        **How to read it:** the swing strategy waits for specific setups, so it is often in
        cash. A low number means your money sits idle most of the time, which is one reason it
        can trail buy and hold in a rising market. Idle cash could earn deposit interest,
        which the backtest does not add.
        """,
    ),
    "bt_vs_bh": _t(
        "vs buy and hold",
        """
        **What it is:** the strategy's total return minus buy and hold's total return, in
        percentage points.

        **How to read it:** positive (green) means the strategy beat simply holding the
        stock. Negative (red) means holding would have made more.
        """,
    ),
    "bt_equity": _t(
        "Portfolio value",
        """
        **What it is:** the value of the replayed portfolio each day, in PKR. The shaded blue
        line is the strategy, the thin grey line is buy and hold with the same starting cash.

        **How to read it:** a line that climbs steadily is better than one that jumps. Flat
        stretches are times when the strategy was in cash. Where the blue line is above the
        grey line, the strategy was ahead at that point. Drag to zoom, double-click to reset.
        """,
    ),
    "bt_drawdown": _t(
        "Drawdown",
        """
        **What it is:** how far the strategy's portfolio was below its highest value so far,
        on each day.

        **How to read it:** 0% means it was at a new high. The deeper the red area, the bigger
        the loss from the peak at that time. Long red stretches mean a long wait to recover.
        """,
    ),
    "bt_trades": _t(
        "Trades",
        """
        **What it is:** every trade the replay made.

        For the **swing strategy** each row is one position: entry and exit dates, shares,
        entry price, stop price, average exit price, sessions held, profit or loss after costs
        in PKR, the result in R, and why it was closed (stop, target, trailing stop and so
        on).

        For the **other strategies** each row is one buy or sell with its price, costs and net
        value.

        **Export trades** saves the list as a spreadsheet file.
        """,
    ),
    "bt_universe": _t(
        "Same rules on every stock",
        """
        **What it is:** the same swing settings replayed on every tracked stock that has
        enough history.

        **How to read it:** it shows whether the rules work broadly or only on one stock. A
        rule that wins on one stock but loses on most is probably luck. Compare **Return** with
        **Buy and hold**, and **Max drawdown** with **B&H drawdown**, row by row.
        """,
    ),
    # ---------------------------------------------------------------- research
    "rs_overview": _t(
        "Latest research across stocks",
        """
        **What it is:** the newest valid research file for each stock, ranked best first.

        **The columns:** **Stance** and **Upside %** are the research recommendation and its
        estimated gain to fair value. **Sentiment** and **Outlook 1-4w** run from -1 to +1.
        **Risk** is 0 to 100. **Confidence** is how sure the research was.

        **Weighted outlook** is the short-term outlook times its weight: the weight halves
        every 30 days, is multiplied by confidence and is zero after 120 days. So fresh,
        confident, positive research ranks highest. **Stale** is ticked when the research is
        older than 30 days or no longer counts.
        """,
    ),
    "rs_files": _t(
        "Research files for this stock",
        """
        **What it is:** every research file saved for this stock, with its headline scores.

        **How to read it:** **Status OK** files are used in predictions, **INVALID** files are
        ignored (open the list of invalid files above to see why). **Video** says whether a
        research video exists for that date. Pick a file with **Research date** below to read
        it.
        """,
    ),
    "rs_history": _t(
        "Research scores over time",
        """
        **What it is:** how this stock's research scores changed from one research file to the
        next: overall sentiment, news sentiment, and the 1-4 week and 3-12 month outlooks.

        **How to read it:** every score runs from -1 (very negative) to +1 (very positive), and
        the dotted line is neutral. Lines moving up mean the view is improving.
        """,
    ),
    "rs_metrics": _t(
        "Research summary",
        """
        - **Stance:** the research recommendation.
        - **Fair value:** the price the research thinks the stock is worth.
        - **Upside:** how far the fair value is above (or below, if negative) the price.
        - **Sentiment:** overall tone from -1 (very negative) to +1 (very positive).
        - **Risk:** 0 to 100, higher is riskier.
        - **Confidence:** how sure the research was. It also scales how much the research
          counts in predictions.
        """,
    ),
    "rs_weight": _t(
        "How research reaches predictions",
        """
        **What it is:** research scores are added as extra inputs to the prediction model for
        this stock.

        **The weight** is how much this research counts today. It starts at the research
        confidence, halves every 30 days and becomes zero after 120 days. Each past session
        only uses research dated on or before it, so the model never sees the future.

        The line under it says whether the research direction **agrees with** the latest
        signal, **disagrees**, or is neutral.
        """,
    ),
    "rs_details": _t(
        "Research details",
        """
        The tabs show the parts of the research file:
        - **Report:** the full written report.
        - **Scores:** every score in the file.
        - **Catalysts and risks:** things that could move the price up or down.
        - **News, Financials, Events:** the facts the research used and upcoming dates.
        - **Sources:** where each fact came from, with links.
        - **Raw JSON:** the file exactly as stored, plus any warnings.
        """,
    ),
    # ---------------------------------------------------------------- pipeline
    "pl_last_run": _t(
        "Last run",
        """
        **What it is:** the date of the most recent daily update that really ran (test runs are
        ignored).

        **How to read it:** green "Finished" means every step worked. Red means some steps
        failed. The step list below shows which.
        """,
    ),
    "pl_last_ok": _t(
        "Last full success",
        """
        **What it is:** the most recent daily run where all six steps worked in one go.

        **How to read it:** if this date is old while "Last run" is recent, recent runs had
        failures, so some data may be stale.
        """,
    ),
    "pl_prices": _t(
        "Prices up to date",
        """
        **What it is:** how many tracked stocks have prices for the newest session on file.

        **How to read it:** "30 / 30" means all are current. Otherwise the stocks that are
        behind are named. Use the update button in the price table to fetch them.
        """,
    ),
    "pl_macro": _t(
        "Macro data",
        """
        **What it is:** the newest session in the macro data: USD/PKR, Brent oil, KSE-100 and
        State Bank of Pakistan rates.

        **How to read it:** it should match the newest trading day. An older date means the
        macro step failed or the sources had no new data.
        """,
    ),
    "pl_predictions": _t(
        "Predictions logged",
        """
        **What it is:** how many predictions are stored, one per stock per session.

        **Duplicates ignored:** running the update twice for the same session logs a second
        prediction. The dashboard keeps only the newest one and counts the rest here.
        """,
    ),
    "pl_run_now": _t(
        "Run now",
        """
        **What it is:** buttons that start a job in the background on this computer.

        - **Run full pipeline:** all six steps, carrying on past a failed step.
        - **Update all prices:** only fetch the newest prices.
        - **Score predictions:** check past predictions against the closes now on file.

        Progress and the log show at the top of the page. The buttons only work when you open
        the dashboard on the computer it runs on. Windows also runs the update every hour after
        the market closes.
        """,
    ),
    "pl_steps": _t(
        "Latest run, step by step",
        """
        **What it is:** each of the six steps of the newest daily run, in order: prices, news
        and announcements, macro data, features, predictions and the audit.

        **How to read it:** green tick worked, grey dash skipped, red mark failed. The right
        side shows how long the step took, or when it was last rerun on its own. Failed steps
        show their error message below the list.

        **Rerun** runs just that one step again.
        """,
    ),
    "pl_price_table": _t(
        "Price data by stock",
        """
        **What it is:** for each stock, the newest session in its price file.

        **The columns:** **Last session** is the newest date on file, **Status** says "Up to
        date" (green) or "Behind" (amber), and **Has prediction** shows whether a signal
        exists for it.

        If some stocks are behind, a button above the table fetches only those.
        """,
    ),
    "pl_recent": _t(
        "Recent runs",
        """
        **What it is:** the last 30 runs logged in data/pipeline.log, newest first.

        **The columns:** when it **started**, the **result** (Success, Failed, Dry run for a
        test, or "rerun" when only some steps ran), which **steps failed**, and how many
        **seconds** it took.

        A pattern of the same step failing day after day points to a source that changed or
        is blocked.
        """,
    ),
}


_MD = re.compile(r"\*\*|__|`")


def _spoken(line: str) -> str:
    """Spell out symbols a voice would read badly or skip."""
    return (
        _MD.sub("", line)
        .replace("USD/PKR", "the dollar rate")
        .replace("USD / PKR", "U S dollar to rupee")
        .replace("data/pipeline.log", "the pipeline log file")
        .replace("30 / 30", "30 out of 30")
        .replace(" · ", ", ")
        .replace("×", " times ")
        .replace("%", " percent")
        .replace("1-4", "1 to 4")
        .replace("3-12", "3 to 12")
        .replace("0-10", "0 to 10")
        .replace("10-20", "10 to 20")
        .replace("-1 ", "minus 1 ")
        .replace("-1,", "minus 1,")
        .replace("+1", "plus 1")
        .replace("PKR", "rupees")
        .replace("B&H", "buy and hold")
        .replace("(x ATR)", "(times ATR)")
        .replace("/", " or ")
    )


def speech_text(t: HelpTopic) -> str:
    """The topic as plain sentences for the read-aloud voice."""
    title = _spoken(t.title)
    out = [title if title.endswith(("?", "!", ".")) else title + "."]
    for raw in t.body.splitlines():
        line = raw.strip()
        if not line:
            continue
        is_item = line.startswith("- ")
        line = _spoken(line[2:] if is_item else line)
        if is_item and not out[-1].endswith((".", ":", "!", "?")):
            out[-1] += "."
        out.append(line)
    joined = " ".join(out)
    return re.sub(r"\s+", " ", joined).replace(" percent percent", " percent").strip()
