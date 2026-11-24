# PSX Research Video Prompt (for Google Antigravity)

Turns each research report in `data/research/` into an animated, narrated video in the
"broker research note" style. Run it after the research prompt
(`antigravity_research_prompt.md`).

Version 2: every element is animated (charts grow and count up, headings slide in, items
reveal in step with the voice), and the narration uses a natural voice and a script
written for speaking. Version 1 made slideshows (still images plus robotic voice); set
`FORCE: true` once to rebuild those.

Copy everything between the two `====` lines into Antigravity, with the
`D:\Development\PSX_Prediction` folder open as the workspace. Change the settings block
first if you need to.

====================================================================

## Task

You are a motion designer and financial video producer. For each stock research report in
`data/research/`, make an animated explainer video in which every chart, number, heading
and list is animated and timed to a natural, human-sounding narration. It must not look
like a slideshow: something meaningful should move whenever the narrator introduces a new
point, and nothing should sit as a static full slide for more than about 3 seconds.

Build a reusable pipeline for this, so it can be rerun for new reports.

## Settings

```
SYMBOLS: ALL               # ALL, or a comma separated list such as OGDC, PPL
ONLY_LATEST: true          # true = only the newest report per symbol
FORCE: false               # true = rebuild videos that already exist (do this once for v1 videos)
LANGUAGE: English          # English | Urdu (Urdu needs TTS_ENGINE elevenlabs or openai)
TTS_ENGINE: kokoro         # kokoro (free, local) | elevenlabs | openai | edge (last resort)
VOICE: af_heart            # kokoro: af_heart, af_bella, am_michael, bf_emma, bm_george
                           # elevenlabs: a voice name or id; openai: e.g. ash, coral, sage
SPEAKING_RATE: 0.95        # 1.0 = engine default; slightly slower sounds more considered
MUSIC: auto                # auto = use data/research/videos/_assets/music.mp3 if it exists; none = no music
RESOLUTION: 1920x1080
FPS: 30
TARGET_LENGTH: 3-4 minutes
```

## Inputs

- `data/research/<SYMBOL>_<YYYY-MM-DD>.json`: the source of every number, date, score,
  chart value and source. Use its `video_brief` section (hook, story, tone,
  chart_highlights, pronunciations, closing_line) to write the narration. Older files may
  not have `video_brief`; then write the same material yourself from the rest of the JSON.
- `data/research/<SYMBOL>_<YYYY-MM-DD>.md`: wording and context. If it disagrees with the
  JSON, the JSON wins.
- `data/processed/prices/<SYMBOL>.parquet` (read only, may be missing): daily prices from
  the app (`trade_date`, `close`, ...). Use the last 250 sessions up to the research date
  for an animated price-trend chart. Never use rows after the research date.
- Skip a report if its JSON is missing or does not parse, and list it in the final summary.

## Outputs (per report)

```
data/research/videos/<SYMBOL>_<YYYY-MM-DD>.mp4           final video
data/research/videos/<SYMBOL>_<YYYY-MM-DD>.srt           subtitles
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/script.md     narration script by scene and beat
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/timeline.json scenes, beats, timings, animation cues
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/audio/        one WAV per beat + the mixed narration track
data/research/videos/<SYMBOL>_<YYYY-MM-DD>/stills/       QA stills (see self-check)
data/research/videos/index.json                          list of all videos
```

Skip a report whose `.mp4` already exists unless `FORCE` is true.

`index.json` is a list with one entry per video, newest first:
`{"symbol": "OGDC", "research_date": "2026-11-23", "video": "OGDC_2026-11-23.mp4", "subtitles": "OGDC_2026-11-23.srt", "duration_sec": 214.0, "style": "Broker research note (animated)", "language": "English", "voice": "kokoro/af_heart", "created_at": "<ISO timestamp>"}`.
Replace the entry for the same symbol and date; keep the others.

## Ground rules

1. Do not edit anything outside `data/research/videos/`. Do not change the project's
   source code, `pyproject.toml` or its `.venv`. Do not run git commands.
2. Never write API keys into any file. Read them only from environment variables
   (`ELEVENLABS_API_KEY`, `OPENAI_API_KEY`). If the chosen engine's key is missing, stop and
   tell me, instead of silently switching to a worse voice.
3. Say only what the research says. No new facts, no price predictions of your own, no
   number that is not in the JSON (or the price file for the trend chart). Skip a scene
   whose data is all `null` or empty.
4. Every video opens with the research date and "data as of" date, and ends with the
   spoken and on-screen line: "This video summarises AI-assisted research and is not
   financial advice."
5. It runs on Windows 11. Use `pathlib` and UTF-8 everywhere.
6. Music, fonts and images: use only files already on this computer or in
   `data/research/videos/_assets/`. Do not download music or images from the web.

## Tools to use

- **Animation and rendering: Remotion** (React + TypeScript, renders every frame
  deterministically, built for data-driven video). Create the Remotion project in
  `data/research/videos/_studio/`. It needs Node.js 20 or newer: check `node -v`; if Node
  is missing, stop and ask me to run `winget install OpenJS.NodeJS.LTS`. Use the official
  blank template (`npx create-video@latest`), TypeScript, and only Remotion's own packages
  plus `@remotion/google-fonts` (or the Segoe UI system font). Draw charts yourself with SVG
  and Remotion's `interpolate`, `spring` and `Easing`, not with a chart library, so every
  bar and label can be animated. Remotion is free for individuals; mention it in the
  summary if a company licence would apply.
- **Data, narration and audio: Python** in `data/research/videos/_pipeline/` with its own
  venv at `data/research/videos/_pipeline/.venv`. Packages: `pandas`, `pyarrow`,
  `soundfile`, `numpy`, `imageio-ffmpeg` (use `imageio_ffmpeg.get_ffmpeg_exe()` for every
  ffmpeg call), and the TTS package for the chosen engine (`kokoro` for kokoro,
  `elevenlabs` for elevenlabs, `openai` for openai, `edge-tts` for edge). Add
  `faster-whisper` (model `small.en`, CPU) only if the TTS engine gives no word timings.
- If a pipeline or studio already exists from an earlier run, reuse it and fix or extend it.

## Step 1: write the narration for the ear

Write `script.md` first. Each scene is split into **beats**: one beat is one to three
sentences about one on-screen element, and each beat names the element it reveals or
emphasises. Rules:

- Sound like a calm, experienced analyst talking to one person, not a document being read.
  Use contractions, plain words, short and varied sentences, and the occasional natural
  connector ("So", "Here's the thing", "That said"). No "Furthermore", "In conclusion",
  "As we can see", or reading table headers aloud.
- Start with the `video_brief.hook`, follow `video_brief.story` as the spine, and end with
  `video_brief.closing_line` and the disclaimer. Match `video_brief.tone`.
- Speak numbers the way a person would: "two hundred and fifty rupees", "about six and a
  half times earnings", "up twenty percent", "the twenty-fourth of November". Round for
  speech, but the screen shows the exact value.
- Apply every entry in `video_brief.pronunciations`, and spell tickers letter by letter
  ("O G D C"). Expand abbreviations once: EPS, P/E, SBP, MPC, KIBOR.
- Don't narrate every number on screen. Say the one or two that matter and let the
  animation show the rest.
- Use punctuation to shape delivery: commas for short pauses, full stops between ideas,
  an em dash or "..." sparingly for a beat of emphasis. Never write all caps.
- Total spoken length about 420 to 560 words for 3 to 4 minutes.

## Step 2: make the voice sound human

- Generate audio **one beat at a time** (not one long block, not word by word), so each
  beat has natural intonation and its own timing.
- kokoro: `KPipeline(lang_code="a")` for American voices, `"b"` for British; `speed` =
  `SPEAKING_RATE`; 24 kHz WAV. Keep its per-token timestamps for captions if available.
- elevenlabs: model `eleven_multilingual_v2`, stability about 0.45, similarity 0.8, style
  about 0.2, speaker boost on; use the "with timestamps" endpoint for character timings.
- openai: model `gpt-4o-mini-tts` with `instructions` such as "Calm, warm, confident
  financial analyst. Natural pace, light emphasis on key numbers, slight pause before the
  main point." Ask for WAV.
- edge: only as a last resort; then say so in the summary because it will sound robotic.
- Between beats insert 250 to 400 ms of silence, 700 ms between scenes, and 500 ms before
  the most important number of a scene. Trim leading and trailing silence from each beat
  first so the pauses are controlled.
- Mix: join beats into one narration track, normalise loudness with ffmpeg `loudnorm`
  (I=-16, TP=-1.5, LRA=11). If `MUSIC` is on and the music file exists, loop it under the
  voice at about -26 dB with sidechain ducking (`sidechaincompress`), fade it in over 2 s
  and out over 3 s.
- Listen-check proxy: if any beat comes out faster than 185 words per minute or has an
  obvious mispronunciation you can detect from the transcript (run faster-whisper on the
  beat and compare), regenerate that beat with adjusted punctuation or spelling.

## Step 3: build the timeline

From the beat audio durations, write `timeline.json`: for each scene, its start frame and
length; for each beat, its start frame, length, text, caption words with timings, and the
animation cue (which element enters or is emphasised at that moment). Remotion reads this
file through `calculateMetadata`, so the video length always follows the audio. Each
element starts animating 3 to 6 frames **before** the narrator mentions it, never after.

## Step 4: animate in the house style

**Style name:** Broker research note (animated)

Look (same as the research note): white background `#FFFFFF`, navy headings `#0B2545`,
teal accent `#13808B`, grey text `#4A4F55`, positive green `#1E8E3E`, negative red
`#C62828`, neutral amber `#B7791F`, light grey `#E6E8EB` for tracks and peers. Segoe UI
(Semibold for headings), tabular figures for every number. 96 px side margins, thin navy
top bar with "<SYMBOL> | <Company> | Research <date>" and the chapter name, footer line
"AI-assisted research, data as of <date> | Not financial advice" in 22 px grey, and a thin
teal progress bar along the bottom edge.

Motion language (use these everywhere, consistently):

- Easing: `Easing.bezier(0.22, 1, 0.36, 1)` for entrances; springs with damping about 200
  (no bounce) for bars and cards. Entrances last 12 to 24 frames.
- **Headings**: slide up 24 px and fade in, then a teal underline wipes left to right.
  Chapter label in the top bar cross-fades.
- **Body text and bullets**: each line fades and rises 12 px, staggered 6 frames, revealed
  on the beat that mentions it. Key phrases get a soft teal highlighter sweep behind them
  when spoken.
- **Numbers**: count up from 0 (or from the previous value) over 20 to 30 frames to the
  exact value with correct decimals, units and sign; then a brief scale pulse (1.0 to
  1.06 to 1.0) when the narrator says it.
- **Bar charts** (scores, valuation vs peers, earnings, dividends): bars grow from the
  baseline with a spring, staggered 4 frames left to right; the value label counts up and
  rides on the end of each bar; the stock's bar is teal and peers light grey; the bar named
  in `chart_highlights` gets a glow and its label turns bold when it is narrated; the
  others dim to 50% opacity during that moment. Axes and gridlines draw in first (8 frames).
- **Signed scores (-1 to 1)**: diverging bars from a centre zero line, green to the right,
  red to the left, with the zero line drawn first.
- **Price trend** (from the price file): the line draws left to right
  (`strokeDashoffset`), the area under it fades in afterwards, the last point gets a pulsing
  dot with the last close counting up beside it, and the 52-week high and low lines slide in
  with labels.
- **52-week range**: the track draws, then a marker slides from the low end to the last
  close, with low, high and last labels counting up.
- **Gauges** (sector outlook, confidence): an arc sweeps to the value while the number
  counts up.
- **Tables** (earnings, analyst views): header fades in, rows slide in from 16 px below,
  staggered 5 frames; the latest row gets a light teal background.
- **Cards** (catalysts, risks): slide in and scale from 0.96 to 1 with a 6-frame stagger;
  direction arrows draw; probability shown as a small bar that fills.
- **Risk grid** (severity vs likelihood): grid lines draw, cells tint by severity, then
  each risk drops in as a labelled dot on its beat.
- **Events timeline**: the line draws, then pins pop in by date with a small spring;
  estimated dates use a dashed pin and "est." label.
- **Stance badge**: rotates in from 8 degrees with a spring and a colour wash
  (green BUY / STRONG_BUY, amber HOLD, red SELL / STRONG_SELL).
- **Scene transitions**: 15-frame cross-fade with a slight horizontal push (40 px); never
  a hard cut except into the sources list.
- **Keep it alive, not busy**: during long beats add a slow camera drift (scale 1.00 to
  1.02 over the scene) on the main chart; never loop flashy effects; nothing moves after
  it lands except the drift and highlight emphasis. Every element stays fully readable for
  at least 2 seconds before the scene leaves.
- **Captions**: two lines at the bottom, 34 px, white text on a navy panel at 85% opacity,
  the current word highlighted teal (word timings from TTS or faster-whisper). Also write
  the same captions to the `.srt` file (cues of at most 42 characters per line, 2 lines).

## Scenes (in this order)

| # | Scene | Elements and their animation | Narration beats |
|---|---|---|---|
| 1 | Cold open | Hook line types on word by word, then logo-style symbol card with company, sector, research and data dates | `video_brief.hook`, then introduce the stock and date |
| 2 | Verdict | Stance badge, fair value and upside count up, last close, 3 takeaways reveal one per beat | Stance and why, one beat per takeaway |
| 3 | Scores | Diverging bars for the four -1..1 scores, then horizontal bars for the 0..100 scores, confidence gauge | The 2 or 3 scores that stand out (use `chart_highlights`) |
| 4 | Price | Price-trend line (if price file exists), 52-week range marker | Where the price is versus its year |
| 5 | Valuation | P/E, P/B and dividend yield vs peers as grouped bars | Cheap or expensive vs peers |
| 6 | Results | Earnings bars by period, latest period table, beat / miss tag | Latest results and the YoY change |
| 7 | Dividends | Payout history bars, yield and payout ratio counters, next book closure pin | Payout record and yield |
| 8 | Sector & macro | Sector outlook gauge, key drivers reveal one per beat | What is driving the sector now |
| 9 | Recent news | Up to 5 headline cards with date and a sentiment dot that fills green / amber / red | The 2 or 3 stories that matter |
| 10 | What to watch | Events timeline for the next 90 days | Dates to watch |
| 11 | Catalysts | Up to 4 cards with direction arrows and probability bars | Biggest upside and downside triggers |
| 12 | Risks | Severity vs likelihood grid with risk dots | Top 2 or 3 risks |
| 13 | Analysts | Table of broker, date, rating and target, with target-price bars vs last close | Consensus and range, if any |
| 14 | Wrap-up | Takeaways recap, score strip, disclaimer card | `video_brief.closing_line`, then the disclaimer |
| 15 | Sources | Numbered source list (publisher, date), 5 seconds, hard cut in, no narration | – |

## Step 5: render

- Render with Remotion: H.264, CRF 18, `yuv420p`, `FPS`, `RESOLUTION`, AAC 192 kbps,
  using the mixed narration (+ music) track. Use `--concurrency` of half the CPU cores.
- Write the `.srt`, then update `index.json`.

## Final self-check (do this before you finish)

1. For each new video, render stills with `npx remotion still` at three moments of three
   different scenes (mid-animation, the moment a highlight is narrated, and the settled
   end state) into `stills/`, open them, and check: nothing overlaps or is cut off, numbers
   match the JSON exactly, labels sit on their bars, captions do not cover charts.
2. Confirm with ffprobe (from imageio-ffmpeg, or `ffmpeg -i <file> -f null -`) that the
   video has one video and one audio stream, the duration is within the target length,
   and narration loudness is about -16 LUFS.
3. Confirm the disclaimer is spoken and shown, and no scene sits with nothing moving for
   more than 3 seconds while narration is playing.
4. `index.json` parses as JSON and lists every video in the folder.
5. Reply with a table: symbol, research date, video path, duration, voice used, scenes
   skipped (no data), and any problems or beats you had to regenerate.

====================================================================
