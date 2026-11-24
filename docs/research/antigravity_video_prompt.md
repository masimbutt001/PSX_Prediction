# PSX Research Video Prompt (for Google Antigravity)

Turns each research report in `data/research/` into a narrated presentation video.
Run it after the research prompt (`antigravity_research_prompt.md`).

Copy everything between the two `====` lines into Antigravity, with the
`D:\Development\PSX_Prediction` folder open as the workspace. Change the
settings block first if you need to.

====================================================================

## Task

You are a financial video producer. For each stock research report in
`data/research/`, make a narrated presentation video (slides + voiceover +
subtitles) in the house style described below. Build a small reusable Python
pipeline to do it, so the same pipeline can be rerun later for new reports.

## Settings

```
SYMBOLS: ALL              # ALL, or a comma separated list such as OGDC, PPL
ONLY_LATEST: true         # true = only the newest report per symbol
FORCE: false              # true = rebuild videos that already exist
LANGUAGE: English         # English | Urdu | English with Urdu subtitles
VOICE: en-GB-RyanNeural   # Urdu: ur-PK-AsadNeural (male) or ur-PK-UzmaNeural (female)
RESOLUTION: 1920x1080
TARGET_LENGTH: 3-4 minutes
```

## Inputs

- `data/research/<SYMBOL>_<YYYY-MM-DD>.md` – the report to present.
- `data/research/<SYMBOL>_<YYYY-MM-DD>.json` – the same research as structured data.
  **Take every number, date, score, chart value and source from the JSON**, and use the
  `.md` for wording and context. If the two disagree, the JSON wins.
- Skip a report if its JSON is missing or does not parse, and list it in the final summary.

## Outputs (per report)

```
data/research/videos/<SYMBOL>_<YYYY-MM-DD>.mp4          final video
data/research/videos/<SYMBOL>_<YYYY-MM-DD>.srt          subtitles
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/script.md    narration script, one section per scene
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/scenes.json  scene list (see below)
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/slides/      one PNG per scene
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/audio/       one MP3 per scene
data/research/videos/index.json                         list of all videos (see below)
```

Skip a report whose `.mp4` already exists unless `FORCE` is true.

`scenes.json` is a list of objects:
`{"id": 1, "title": "...", "slide": "slides/01_title.png", "audio": "audio/01_title.mp3", "duration_sec": 12.4, "narration": "..."}`

`index.json` is a list with one entry per video, newest first:
`{"symbol": "OGDC", "research_date": "2026-11-23", "video": "OGDC_2026-11-23.mp4", "subtitles": "OGDC_2026-11-23.srt", "duration_sec": 214.0, "style": "<style name>", "language": "English", "created_at": "<ISO timestamp>"}`.
Update it in place: replace the entry for the same symbol and date, keep the others.

## Ground rules

1. Do not edit any file outside `data/research/videos/`. Do not change the project's
   source code, `pyproject.toml` or its `.venv`. Do not run git commands.
2. Put the pipeline in `data/research/videos/_pipeline/` (for example `make_videos.py`,
   `slides.py`, `narration.py`, `assemble.py`, `style.py`) and give it its own virtual
   environment at `data/research/videos/_pipeline/.venv`. If the pipeline already exists,
   reuse it and only fix or extend it.
3. Python packages for that venv: `edge-tts`, `pillow`, `matplotlib`, `imageio-ffmpeg`
   (it ships an ffmpeg binary, so nothing needs a system install), and `pyttsx3` as an
   offline fallback voice. Use `imageio_ffmpeg.get_ffmpeg_exe()` for every ffmpeg call.
4. Say only what the research says. No new facts, no price predictions of your own, and no
   numbers that are not in the JSON. Skip a scene whose data is all `null` or empty.
5. Every video opens with the research date and "data as of" date, and ends with:
   "This video summarises AI-assisted research and is not financial advice."
6. It runs on Windows 11. Use `pathlib`, UTF-8 everywhere, and fonts from
   `C:\Windows\Fonts` (Segoe UI: `segoeui.ttf`, `segoeuib.ttf`, `seguisb.ttf`; for Urdu text
   use `Nirmala UI` or another installed font that renders Urdu, and check it renders).

## Scenes (in this order)

| # | Scene | What to show | Narration |
|---|---|---|---|
| 1 | Title | Symbol, company name, sector, research date, data as of | Introduce the stock and the date |
| 2 | Verdict | Stance badge, fair value, upside %, last close, 3 key takeaways | Stance and the reason in 2-3 sentences |
| 3 | Scores | Horizontal bars: sentiment, news sentiment, outlook short and medium (-1 to 1, centred on 0); fundamental, valuation, growth, dividend, financial health, governance, risk (0 to 100); confidence | The 3 most notable scores and what they mean |
| 4 | Price & valuation | 52-week range bar with last close marked; P/E, P/B, dividend yield vs peers (bar chart) | Cheap or expensive vs peers and history |
| 5 | Financial results | Bar chart of EPS or net income for recent periods; latest period table | Latest results and YoY change, beat or miss |
| 6 | Dividends | Payout history bars, yield, payout ratio, next book closure | Payout record and yield |
| 7 | Sector & macro | Sector outlook gauge, key drivers as bullets | What drives this sector now |
| 8 | Recent news | Up to 5 items: date, headline, coloured sentiment dot | The 2-3 most important items |
| 9 | Upcoming events | Timeline of the next 90 days, estimated dates marked "est." | What to watch and when |
| 10 | Catalysts | Up to 4 cards: title, direction arrow, impact, probability | The biggest upside and downside triggers |
| 11 | Risks | 3x3 severity vs likelihood grid with risks placed in it | The top 2-3 risks |
| 12 | Analyst views | Table: broker, date, rating, target price | Consensus and range, if any |
| 13 | Wrap-up | Takeaways again, scores summary, disclaimer | Short close plus the disclaimer |
| 14 | Sources | Numbered source list (publisher, date); 5 seconds, no narration needed beyond "Sources are listed on screen." | – |

Narration per scene: 30 to 90 words, plain spoken English (or Urdu), short sentences.
Speak numbers naturally: "Rs 250.4", "six point six times earnings", "up 20 percent",
"the twenty-fourth of November". Spell out abbreviations once (EPS, P/E, SBP, MPC).

## House style

**Style name:** Broker research note

- Look: like a printed broker research note brought to screen. White background
  (`#FFFFFF`), navy headings (`#0B2545`), teal accent (`#13808B`), grey text (`#4A4F55`),
  positive green `#1E8E3E`, negative red `#C62828`, neutral amber `#B7791F`.
- Layout: 1920x1080. Thin navy top bar with "<SYMBOL> | <Company> | Research <date>" on the
  left and the scene title on the right; 96 px side margins; footer line with
  "Source: AI-assisted research, data as of <date> | Not financial advice" in 22 px grey.
- Type: headings Segoe UI Semibold 56 px, body 34 px, table text 28 px, numbers in bold.
  At most 6 bullets or one chart plus 3 bullets per slide.
- Charts: matplotlib, flat style, no gridlines except light grey horizontal ones, value
  labels on bars, teal for the stock, light grey for peers, green/red for signed scores.
- Motion: none inside a slide; 0.4 s crossfade between scenes.
- Voice and pace: calm and formal, around 150 words per minute (`edge-tts` rate "-5%").
- Subtitles: soft subtitle track in the MP4 plus the `.srt` file, not burned in.
- Length: 3 to 4 minutes.

## How to build each video

1. Read the JSON and the `.md`; decide which scenes have data.
2. Write `script.md` and the narration for each scene.
3. Render each slide to PNG at the full resolution.
4. Make one MP3 per scene with `edge-tts` (voice and rate from the settings and style).
   If `edge-tts` fails (no internet), use `pyttsx3` with an installed Windows voice and
   say so in the final summary.
5. Scene duration = audio length + 0.8 seconds (5 seconds for the sources scene).
6. Build subtitles from the narration, split into lines of at most 42 characters and
   2 lines per cue, timed across each scene's audio.
7. Join the scenes with ffmpeg: H.264 (`libx264`, `-pix_fmt yuv420p`, 30 fps, CRF 20),
   AAC audio 160 kbps, crossfades as the style says, subtitles as the style says.
8. Write `scenes.json`, then update `index.json`.

## Final self-check (do this before you finish)

1. Every new `.mp4` plays: run ffprobe (from the imageio-ffmpeg folder, or
   `ffmpeg -i <file> -f null -`) and confirm it has one video and one audio stream and a
   duration within the target length (or explain why it is shorter).
2. Open two random slides of each video as images and check text is not cut off or
   overlapping and numbers match the JSON.
3. Confirm the disclaimer is spoken and shown.
4. `index.json` parses as JSON and lists every video in the folder.
5. Reply with a table: symbol, research date, video path, duration, scenes skipped
   (no data), and any problems.

====================================================================
