"use strict";
// All text from the database/crawled sources is untrusted: it is only ever inserted with textContent / attributes.
const $ = (id) => document.getElementById(id);
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c != null) n.append(c.nodeType ? c : document.createTextNode(String(c)));
  return n;
}
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);
const fmtT = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const fmtDur = (s) => (s >= 3600 ? `${(s / 3600).toFixed(1)} h` : s >= 60 ? `${(s / 60).toFixed(1)} min` : `${Math.round(s)} s`);
async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch (_) { /* not json */ }
  if (!r.ok) throw new Error((body && (body.detail || body.error)) || `${r.status} ${r.statusText}`);
  return body;
}

// ---------------------------------------------------------------- tabs
let view = "find";
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
function show(v) {
  view = v;
  document.querySelectorAll("nav button").forEach((b) => {
    const on = b.dataset.view === v;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on);
  });
  document.querySelectorAll(".view").forEach((s) => (s.hidden = s.id !== "view-" + v));
  if (v === "db") { loadVoices(true); loadSources(); }
  if (v === "crawl") pollCrawl();
}

async function refreshStats() {
  try {
    const s = await api("/api/stats");
    $("dbline").textContent = s.clips
      ? `Database: ${s.voices.toLocaleString()} voices · ${s.clips.toLocaleString()} singing clips · ${fmtDur(s.singing_seconds)} of singing · ${s.sources.toLocaleString()} videos searched`
      : "The database is empty — open the Crawler tab to start filling it.";
  } catch (_) { /* ignore */ }
}

// ---------------------------------------------------------------- record / upload
let blob = null, recorder = null, stream = null, audioCtx = null, timer = null, t0 = 0;

function setBlob(b, label) {
  blob = b;
  $("go-btn").disabled = !b;
  $("file-name").textContent = label || "";
}
$("file-in").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (f) { $("rec-audio").hidden = true; setBlob(f, `${f.name} (${(f.size / 1e6).toFixed(1)} MB)`); }
});
const drop = document.querySelector(".drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (f) { $("rec-audio").hidden = true; setBlob(f, `${f.name} (${(f.size / 1e6).toFixed(1)} MB)`); }
});

function pickMime() {
  for (const m of ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"])
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return "";
}
async function startRec() {
  if (!navigator.mediaDevices || !window.MediaRecorder) { setStatus("This browser can't record audio — upload a file instead.", "bad"); return; }
  try {
    // singing: no echo cancellation / noise suppression / AGC, they distort the voice
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) { setStatus("Microphone unavailable: " + e.message, "bad"); return; }
  const mime = pickMime();
  const chunks = [];
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  recorder.onstop = () => {
    const type = recorder.mimeType || mime || "audio/webm";
    const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
    const b = new File(chunks, `recording.${ext}`, { type });
    $("rec-audio").src = URL.createObjectURL(b);
    $("rec-audio").hidden = false;
    setBlob(b, "");
    stream.getTracks().forEach((t) => t.stop());
    audioCtx && audioCtx.close();
    $("meter-fill").style.width = "0";
  };
  // level meter
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const an = audioCtx.createAnalyser(); an.fftSize = 1024;
  audioCtx.createMediaStreamSource(stream).connect(an);
  const buf = new Uint8Array(an.fftSize);
  (function loop() {
    if (!recorder || recorder.state === "inactive") return;
    an.getByteTimeDomainData(buf);
    let peak = 0; for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
    $("meter-fill").style.width = Math.min(100, (peak / 128) * 140) + "%";
    requestAnimationFrame(loop);
  })();
  recorder.start(250);
  t0 = Date.now();
  timer = setInterval(() => ($("rec-time").textContent = fmtT((Date.now() - t0) / 1000)), 200);
  $("rec-btn").textContent = "■ Stop";
  $("rec-btn").classList.add("rec");
  setStatus("Recording…");
}
function stopRec() {
  clearInterval(timer);
  recorder.stop();
  $("rec-btn").textContent = "● Record again";
  $("rec-btn").classList.remove("rec");
  setStatus("");
}
$("rec-btn").addEventListener("click", () => (recorder && recorder.state === "recording" ? stopRec() : startRec()));

function setStatus(msg, cls) {
  const s = $("status");
  s.textContent = msg;
  s.style.color = cls === "bad" ? "var(--bad)" : "";
}

// ---------------------------------------------------------------- match
$("go-btn").addEventListener("click", async () => {
  if (!blob) return;
  $("go-btn").disabled = true;
  setStatus("");
  $("status").append(el("span", { class: "spinner" }), "Listening for singing and comparing voices…");
  const fd = new FormData();
  fd.append("file", blob, blob.name || "recording.webm");
  try {
    const res = await api("/api/match", { method: "POST", body: fd });
    setStatus("");
    renderResults(res);
  } catch (e) {
    setStatus(e.message, "bad");
  } finally {
    $("go-btn").disabled = false;
  }
});

function attribution(c) {
  const parts = [];
  const page = safeUrl(c.page_url);
  parts.push(page ? el("a", { href: page, target: "_blank", rel: "noopener noreferrer", text: c.title || "source" }) : (c.title || "source"));
  if (c.author) parts.push(` · ${c.author}`);
  if (c.license) {
    const lu = safeUrl(c.license_url);
    parts.push(" · ", lu ? el("a", { href: lu, target: "_blank", rel: "noopener noreferrer", text: c.license }) : c.license);
  }
  parts.push(` · ${fmtT(c.start)}–${fmtT(c.end)} of the original`);
  return el("div", { class: "attr" }, parts);
}

function matchCard(m, rank) {
  const video = el("video", { controls: true, preload: "metadata", playsinline: true, src: m.clips[0].url });
  const attr = el("div");
  const showClip = (i, chipEls) => {
    video.src = m.clips[i].url; video.load();
    attr.replaceChildren(attribution(m.clips[i]));
    chipEls.forEach((c, j) => c.classList.toggle("on", i === j));
  };
  const chips = [];
  m.clips.forEach((c, i) => chips.push(el("button", { class: "chip", type: "button", text: `clip ${i + 1} · ${Math.round(c.similarity * 100) / 100}`, onclick: () => showClip(i, chips) })));
  const more = el("button", { class: "chip", type: "button", text: `all ${m.n_clips} clips of this voice`, onclick: async () => {
    const r = await api(`/api/voices/${m.voice_id}?limit=60`);
    const list = el("div", { class: "grid" }, r.clips.map((c) => el("div", { class: "vcard" },
      el("video", { controls: true, preload: "none", src: c.url }), attribution(c))));
    more.replaceWith(list);
  } });
  showClip(0, chips);
  return el("div", { class: "match" },
    el("div", {}, video, attr),
    el("div", {},
      el("div", { class: "rank", text: rank === 1 ? "Closest match" : `#${rank}` }),
      el("h3", { text: m.label || `Voice ${m.voice_id}` }),
      el("div", { class: "pct", text: `${m.percent}%` }),
      el("div", { class: "bar-bg" }, el("div", { class: "bar-fg", style: `width:${Math.min(100, m.percent)}%` })),
      el("div", { class: "muted small", text: `cosine similarity ${m.best.toFixed(3)} · ${m.n_clips} clips · ${fmtDur(m.total_sec)} of singing in the database` }),
      el("div", { class: "chips" }, chips, more)));
}

function renderResults(res) {
  const box = $("results");
  box.hidden = false;
  box.replaceChildren();
  const head = el("div", { class: "card" }, el("h2", { text: "Your sound-alikes" }));
  const q = res;
  head.append(el("p", { class: "muted", text: q.singing_detected
    ? `Found ${q.singing_seconds}s of singing in your ${fmtDur(q.duration)} ${q.voices.length > 1 ? `with ${q.voices.length} distinct voices` : "recording"}.`
    : `We couldn't clearly detect singing, so we compared the whole ${fmtDur(q.duration)}.` }));
  if (!q.singing_detected) head.append(el("div", { class: "alert warn", text: "No clear singing was detected — results may be unreliable. Try singing louder and closer to the microphone." }));
  if (!q.database.clips) head.append(el("div", { class: "alert bad", text: "The database has no clips yet. Open the Crawler tab and let it run first." }));
  q.results.forEach((r, qi) => {
    if (q.results.length > 1) head.append(el("h3", { text: `Voice ${qi + 1} in your upload (${Math.round(r.query_voice_seconds)}s)` }));
    r.matches.forEach((m, i) => head.append(matchCard(m, i + 1)));
  });
  box.append(head);
  head.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- database tab
let voiceOffset = 0;
async function loadVoices(reset) {
  if (reset) { voiceOffset = 0; $("voices").replaceChildren(); }
  const r = await api(`/api/voices?limit=24&offset=${voiceOffset}`);
  r.voices.forEach((v) => $("voices").append(el("div", { class: "vcard" },
    v.clip ? el("video", { controls: true, preload: "none", src: v.clip.url }) : null,
    el("strong", { text: v.label || `Voice ${v.id}` }),
    el("div", { class: "muted small", text: `${v.n_clips} clips · ${fmtDur(v.seconds)}` }),
    v.clip ? attribution(v.clip) : null)));
  voiceOffset += r.voices.length;
  $("more-voices").hidden = voiceOffset >= r.total;
  if (!r.total) $("voices").append(el("p", { class: "muted", text: "No voices yet." }));
}
$("more-voices").addEventListener("click", () => loadVoices(false));
async function loadSources() {
  const st = $("src-filter").value;
  const r = await api(`/api/sources?limit=100${st ? "&status=" + st : ""}`);
  const tb = document.querySelector("#sources tbody");
  tb.replaceChildren(...r.sources.map((s) => el("tr", {},
    el("td", {}, safeUrl(s.page_url) ? el("a", { href: s.page_url, target: "_blank", rel: "noopener noreferrer", text: s.title || s.ref }) : (s.title || s.ref)),
    el("td", { text: s.kind }), el("td", { text: s.license || "" }),
    el("td", {}, el("span", { class: `tag ${s.status}`, text: s.status, title: s.error || "" })),
    el("td", { text: s.n_voices ?? "" }), el("td", { text: s.n_clips ?? "" }))));
}
$("src-filter").addEventListener("change", loadSources);

// ---------------------------------------------------------------- crawler tab
let crawlRunning = false;
async function pollCrawl() {
  if (view !== "crawl") return;
  try {
    const s = await api("/api/crawl/status");
    crawlRunning = s.running;
    $("crawl-toggle").textContent = s.running ? "Stop crawling" : "Start crawling";
    const q = s.sources_by_status || {};
    const kv = [["Videos found", s.sources], ["Queued", q.queued || 0], ["Analysing", q.working || 0],
      ["With singing", q.done || 0], ["No singing", q.no_singing || 0], ["Failed", q.failed || 0],
      ["Voices", s.voices], ["Clips", s.clips], ["Singing", fmtDur(s.singing_seconds)],
      ["Searches done", `${s.queries_exhausted}/${s.queries_total}`]];
    $("crawl-kv").replaceChildren(...kv.map(([k, v]) => el("div", {}, el("dt", { text: k }), el("dd", { text: typeof v === "number" ? v.toLocaleString() : v }))));
    const lines = s.events.map((e) => `${new Date(e.t * 1000).toLocaleTimeString()}  ${e.msg}`);
    if (s.discovering) lines.push(`… searching: ${s.discovering}`);
    s.active.forEach((a) => lines.push(`… analysing: ${a}`));
    $("events").textContent = lines.join("\n") || "Idle.";
    refreshStats();
  } catch (_) { /* ignore */ }
  setTimeout(pollCrawl, 2500);
}
$("crawl-toggle").addEventListener("click", async () => {
  if (crawlRunning) { await api("/api/crawl/stop", { method: "POST" }); return; }
  const sources = [];
  if ($("src-zenodo").checked) sources.push("zenodo");
  if ($("src-commons").checked) sources.push("commons");
  await api("/api/crawl/start", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sources, workers: Number($("workers").value) || 2 }) });
});
$("url-add").addEventListener("click", async () => {
  const u = $("url-in").value.trim();
  if (!u) return;
  const r = await api("/api/crawl/urls", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ urls: [u] }) });
  $("url-in").value = "";
  $("ingest-msg").textContent = r.added ? "Queued." : "That URL was already known.";
});
$("ingest-in").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f, f.name);
  $("ingest-msg").textContent = "Uploading…";
  try { await api("/api/ingest", { method: "POST", body: fd }); $("ingest-msg").textContent = "Queued for analysis."; }
  catch (err) { $("ingest-msg").textContent = err.message; }
});

refreshStats();
setInterval(refreshStats, 15000);
