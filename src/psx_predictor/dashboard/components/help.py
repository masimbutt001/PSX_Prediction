"""Explanation popups: a small (?) button on each card, chart and table that opens a plain
language explanation, with a Listen button that reads it aloud.

The text lives in `dashboard/help_text.py`. Reading aloud uses the browser's own speech
engine (Web Speech API), so it is free, needs no internet for the built-in Windows voices
and nothing is installed. Microsoft Edge offers the most natural voices on Windows 11
("Microsoft ... Online (Natural)"); Chrome offers its Google voices.
"""

from contextlib import contextmanager
from typing import Iterator, Optional

import streamlit as st

from psx_predictor.dashboard.help_text import HELP, HelpTopic, speech_text
from psx_predictor.dashboard.theme import text

_KEY_PREFIX = "psxhelp-"


def topic(key: str) -> HelpTopic:
    """The help topic for `key`; a KeyError names the missing key."""
    try:
        return HELP[key]
    except KeyError as exc:  # pragma: no cover - guarded by a unit test
        raise KeyError(f"No help text for '{key}'. Add it to dashboard/help_text.py.") from exc


def listen_bar(key: str) -> str:
    """HTML for the Listen button and the voice and speed pickers of one topic."""
    say = text(speech_text(topic(key)))
    return (
        '<div class="psx-listenbar">'
        f'<button type="button" class="psx-listen" data-say="{say}" '
        'title="Read this explanation aloud">'
        '<span class="psx-listen-icon" aria-hidden="true">▶</span>'
        '<span class="psx-listen-label">Listen</span></button>'
        '<select class="psx-rate" aria-label="Reading speed" title="Reading speed">'
        '<option value="0.85">0.85×</option><option value="1" selected>1×</option>'
        '<option value="1.15">1.15×</option><option value="1.3">1.3×</option></select>'
        '<select class="psx-voice" aria-label="Voice" title="Voice">'
        '<option value="">Best voice</option></select>'
        "</div>"
    )


def help_button(key: str) -> None:
    """A (?) button that opens the explanation for `key` in a popover."""
    t = topic(key)
    with st.popover(
        "",
        icon=":material/help:",
        type="tertiary",
        help=f"What is “{t.title}”? Click to read or listen.",
        key=f"psxq-{key}",
    ):
        st.markdown(f"**{t.title}**")
        st.markdown(t.body)
        st.html(listen_bar(key))


def section_title(title: str, key: str, level: int = 5) -> None:
    """A section heading with its (?) button right after it."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.markdown(f"{'#' * level} {title}", width="content")
        help_button(key)


@contextmanager
def helped(key: str, border: Optional[bool] = None) -> Iterator[None]:
    """A container whose (?) button floats in its top-right corner.

    Use it around KPI cards, metrics and other elements with no heading of their own.
    """
    with st.container(key=f"{_KEY_PREFIX}{key}", border=border):
        help_button(key)
        yield


def inline_help(caption: str, key: str) -> None:
    """A muted caption followed by a (?) button, for tabs and expanders."""
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.caption(caption, width="content")
        help_button(key)


SPEECH_JS = r"""
<script>
(function () {
  const w = window;
  if (w.__psxSpeech) { return; }
  w.__psxSpeech = true;
  const synth = w.speechSynthesis;
  const store = {
    get(k) { try { return w.localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { w.localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };
  let current = null;  // button that is speaking

  function englishVoices() {
    if (!synth) { return []; }
    return synth.getVoices().filter(v => /^en([-_]|$)/i.test(v.lang));
  }
  function score(v) {
    let s = 0;
    if (/natural/i.test(v.name)) s += 100;          // Edge: Microsoft ... Online (Natural)
    if (/online/i.test(v.name)) s += 40;
    if (/google/i.test(v.name)) s += 50;            // Chrome: Google US English
    const nice = /aria|jenny|guy|ava|andrew|emma|brian|sonia|libby|ryan|neerja|prabhat/i;
    if (nice.test(v.name)) s += 15;
    if (/en-(US|GB)/i.test(v.lang)) s += 5;
    if (v.localService) s += 1;
    return s;
  }
  function bestVoice() {
    const voices = englishVoices();
    const saved = store.get("psx_voice");
    const chosen = saved && voices.find(v => v.name === saved);
    if (chosen) { return chosen; }
    return voices.slice().sort((a, b) => score(b) - score(a))[0] || null;
  }
  function fillPickers(root) {
    const voices = englishVoices().slice().sort((a, b) => score(b) - score(a));
    const saved = store.get("psx_voice") || "";
    const rate = store.get("psx_rate") || "1";
    root.querySelectorAll("select.psx-voice").forEach(sel => {
      if (sel.dataset.filled === String(voices.length)) { return; }
      sel.innerHTML = "";
      const auto = document.createElement("option");
      auto.value = "";
      auto.textContent = "Best voice";
      sel.appendChild(auto);
      voices.forEach(v => {
        const o = document.createElement("option");
        o.value = v.name;
        o.textContent = v.name.replace(/^Microsoft\s+/i, "").replace(/\s*-\s*English.*$/i, "");
        sel.appendChild(o);
      });
      sel.value = voices.some(v => v.name === saved) ? saved : "";
      sel.dataset.filled = String(voices.length);
    });
    root.querySelectorAll("select.psx-rate").forEach(sel => {
      if (!sel.dataset.filled) { sel.value = rate; sel.dataset.filled = "1"; }
    });
  }
  function setState(btn, speaking) {
    if (!btn) { return; }
    btn.classList.toggle("psx-speaking", speaking);
    const icon = btn.querySelector(".psx-listen-icon");
    const label = btn.querySelector(".psx-listen-label");
    if (icon) icon.textContent = speaking ? "■" : "▶";
    if (label) label.textContent = speaking ? "Stop" : "Listen";
  }
  function stop() {
    if (synth) synth.cancel();
    setState(current, false);
    current = null;
  }
  // Long utterances get cut off in Chrome, so speak sentence by sentence.
  function chunks(say) {
    const parts = say.match(/[^.!?]+[.!?]*\s*/g) || [say];
    const out = [];
    let buf = "";
    parts.forEach(p => {
      if ((buf + p).length > 220 && buf) { out.push(buf.trim()); buf = ""; }
      buf += p;
    });
    if (buf.trim()) out.push(buf.trim());
    return out;
  }
  function speak(btn) {
    const say = btn.dataset.say || "";
    if (!synth || !w.SpeechSynthesisUtterance) {
      const label = btn.querySelector(".psx-listen-label");
      if (label) label.textContent = "Not supported in this browser";
      return;
    }
    stop();
    current = btn;
    setState(btn, true);
    const bar = btn.closest(".psx-listenbar");
    const rateSel = bar && bar.querySelector("select.psx-rate");
    const rate = parseFloat((rateSel && rateSel.value) || store.get("psx_rate") || "1") || 1;
    const voice = bestVoice();
    const list = chunks(say);
    list.forEach((part, i) => {
      const u = new SpeechSynthesisUtterance(part);
      if (voice) { u.voice = voice; u.lang = voice.lang; } else { u.lang = "en-US"; }
      u.rate = rate;
      if (i === list.length - 1) {
        u.onend = () => { if (current === btn) { setState(btn, false); current = null; } };
      }
      u.onerror = () => { if (current === btn) { setState(btn, false); current = null; } };
      synth.speak(u);
    });
  }

  document.addEventListener("click", e => {
    const btn = e.target.closest && e.target.closest("button.psx-listen");
    if (!btn) { return; }
    e.preventDefault();
    e.stopPropagation();
    if (current === btn) { stop(); } else { speak(btn); }
  }, true);
  document.addEventListener("change", e => {
    const t = e.target;
    if (t.matches && t.matches("select.psx-voice")) {
      store.set("psx_voice", t.value);
      document.querySelectorAll("select.psx-voice").forEach(s => { s.value = t.value; });
    } else if (t.matches && t.matches("select.psx-rate")) {
      store.set("psx_rate", t.value);
      document.querySelectorAll("select.psx-rate").forEach(s => { s.value = t.value; });
    }
  }, true);
  if (synth) {
    synth.onvoiceschanged = () => fillPickers(document);
  }
  new MutationObserver(() => {
    const fresh = "select.psx-voice:not([data-filled]), select.psx-rate:not([data-filled])";
    if (document.querySelector(fresh)) {
      fillPickers(document);
    }
    // The explanation was closed or the page changed while reading: stop talking.
    if (current && !document.body.contains(current)) { stop(); }
  }).observe(document.body, { childList: true, subtree: true });
  fillPickers(document);
})();
</script>
"""


def inject_speech() -> None:
    """Load the read-aloud script once per page (it ignores repeat loads)."""
    st.html(SPEECH_JS, unsafe_allow_javascript=True)
