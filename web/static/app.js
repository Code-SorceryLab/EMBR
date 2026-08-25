/* EMBR web demo — fetches a snapshot from the stub-backed server and renders the stage
   plus the five research tabs. No scoring or attribution happens here; the server computes
   every number from the real pipeline and this file only draws it. */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

/* valence in [-1,1] -> a warm(positive)/cool(negative) colour on the affect axis. */
function affectColor(valence) {
  const v = Math.max(-1, Math.min(1, valence || 0));
  return v >= 0
    ? `color-mix(in oklab, var(--pos) ${Math.round(30 + v * 70)}%, var(--parch-mut))`
    : `color-mix(in oklab, var(--neg) ${Math.round(30 + -v * 70)}%, var(--parch-mut))`;
}
const emberRamp = (t) => `color-mix(in oklab, var(--ember) ${Math.round(20 + Math.max(0, Math.min(1, t)) * 80)}%, var(--ink-3))`;

let latest = null;
const settings = { motion: true, typewriter: true, music: false, research: true };
const prefersMotion = () => settings.motion &&
  !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

async function api(path, body) {
  const res = await fetch(path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {});
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

/* ------------------------------------------------------------------ the stage */

let typeTimer = null;

function setPortrait(name) {
  const img = $("#dawn-img");
  const next = `/portraits/${name}.png`;
  if (img.dataset.portrait === name) return;
  img.dataset.portrait = name;
  // crossfade: fade out, swap, fade in (unless motion is off)
  if (prefersMotion()) {
    img.classList.add("swapping");
    setTimeout(() => {
      img.src = next;
      img.alt = `Dawn Whitmore (${name.replace("dawn-", "")})`;
      requestAnimationFrame(() => img.classList.remove("swapping"));
    }, 220);
  } else {
    img.src = next;
    img.alt = `Dawn Whitmore (${name.replace("dawn-", "")})`;
  }
}

function typewrite(node, text) {
  if (typeTimer) { clearInterval(typeTimer); typeTimer = null; }
  if (!text) { node.textContent = ""; return; }
  if (!prefersMotion() || !settings.typewriter) { node.textContent = text; return; }
  node.textContent = "";
  const caret = el("span", "caret", "▏");
  node.appendChild(caret);
  let i = 0;
  typeTimer = setInterval(() => {
    i += 2;
    caret.before(text.slice(i - 2, i));
    if (i >= text.length) { clearInterval(typeTimer); typeTimer = null; caret.remove(); }
  }, 16);
}

function renderStage(s) {
  const stage = s.stage;
  setPortrait(stage.portrait);
  $("#scene").textContent = stage.narration || "";
  typewrite($("#reply"), stage.reply || "");
  $("#watch").textContent = stage.watch_for || "";
  // announce the whole turn once to screen readers, so the typewriter never re-reads partial text
  const announce = [stage.narration, stage.reply, stage.watch_for].filter(Boolean).join(". ");
  if (announce) $("#sr-live").textContent = announce;

  const choices = $("#choices");
  choices.replaceChildren();
  s.choices.forEach((line, idx) => {
    const b = el("button", "choice", line);
    b.type = "button";
    b.style.setProperty("--i", `${idx * 60}ms`);
    b.addEventListener("click", () => say(line));
    choices.appendChild(b);
  });

  const p = s.progress;
  $("#progress").textContent = p.finished
    ? `arc complete · free play (${p.total} scenes played)`
    : `scene ${p.played + 1} of ${p.total}`;
  $("#say-input").placeholder = p.finished
    ? "The arc is done. Ask her anything…"
    : "…or say it in your own words";
}

/* ------------------------------------------------------------- research tabs */

function renderMemories(t) {
  const root = $("#memories");
  root.replaceChildren();
  if (!t.cards.length) {
    root.appendChild(emptyNote(
      "Her store is empty until the arc writes to it. Play a line and the memory it creates appears here, scored."));
    return;
  }
  for (const c of t.cards) {
    const lit = c.rank != null;
    const card = el("div", `mcard ${lit ? "is-lit" : "is-dim"}`);
    const top = el("div", "mcard__top");
    top.appendChild(el("span", "mcard__rank", lit ? `#${c.rank}` : "—"));
    top.appendChild(el("span", "mcard__type", c.event_type));
    if (lit) top.appendChild(el("span", "mcard__score", c.score.toFixed(3)));
    card.appendChild(top);
    card.appendChild(el("p", "mcard__text", c.text));

    const tags = el("div", "mcard__tags");
    const vdot = el("span", "va");
    const dot = el("span", "va__dot"); dot.style.background = affectColor(c.valence);
    vdot.append(dot, `v ${c.valence >= 0 ? "+" : ""}${c.valence.toFixed(2)}  ·  a ${c.arousal.toFixed(2)}`);
    tags.appendChild(vdot);
    card.appendChild(tags);

    if (c.contributions) {
      const total = Object.values(c.contributions).reduce((a, b) => a + Math.abs(b), 0) || 1;
      const bars = el("div", "bars");
      const hues = { relevance: "#5b8fb0", recency: "#8a7bb0", affect: "#e0b341", event_gate: "#7cb08a", mood: "#ea580c" };
      for (const [k, v] of Object.entries(c.contributions)) {
        const seg = el("span");
        seg.style.width = `${(Math.abs(v) / total) * 100}%`;
        seg.style.background = hues[k] || "var(--line-2)";
        seg.title = `${k}: ${v.toFixed(3)}`;
        bars.appendChild(seg);
      }
      card.appendChild(bars);
    }
    root.appendChild(card);
  }
}

function gauge(name, before, after, lo, hi, bipolar) {
  const wrap = el("div", "gauge");
  const label = el("div", "gauge__label");
  label.appendChild(el("span", "gauge__name", name));
  const d = after - before;
  const delta = el("span", `gauge__delta ${d > 0.001 ? "up" : d < -0.001 ? "down" : "flat"}`,
    `${d >= 0 ? "+" : ""}${d.toFixed(3)}`);
  label.appendChild(delta);
  wrap.appendChild(label);

  const track = el("div", `track ${bipolar ? "track--bipolar" : ""}`);
  const pct = (x) => ((x - lo) / (hi - lo)) * 100;
  const fill = el("div", "fill");
  if (bipolar) {
    const mid = pct(0), pos = pct(after);
    fill.style.left = `${Math.min(mid, pos)}%`;
    fill.style.width = `${Math.abs(pos - mid)}%`;
  } else {
    fill.style.left = "0%"; fill.style.width = `${pct(after)}%`;
  }
  fill.style.background = name === "Trust" ? "var(--trust)" : affectColor(bipolar ? after : after - (hi + lo) / 2);
  const ghost = el("div", "ghost"); ghost.style.left = `${pct(before)}%`;
  ghost.title = `before: ${before.toFixed(3)}`;
  track.append(fill, ghost);
  wrap.appendChild(track);

  const scale = el("div", "gauge__scale");
  scale.append(el("span", "", String(lo)), el("span", "", bipolar ? "0" : ""), el("span", "", String(hi)));
  wrap.appendChild(scale);
  return wrap;
}

function circumplex(before, after) {
  // A Russell circumplex: x = valence [-1,1], y = arousal [0,1] mapped to the upper half.
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 100 100");
  const cx = 50, x = (v) => 50 + v * 44, y = (a) => 92 - a * 84;
  const mk = (tag, attrs) => { const n = document.createElementNS(NS, tag); for (const k in attrs) n.setAttribute(k, attrs[k]); return n; };
  svg.append(
    mk("circle", { cx: 50, cy: 50, r: 46, class: "circumplex__ring" }),
    mk("line", { x1: 4, y1: 50, x2: 96, y2: 50, class: "circumplex__axis" }),
    mk("line", { x1: 50, y1: 4, x2: 50, y2: 96, class: "circumplex__axis" }),
  );
  const labels = [["calm", 50, 99], ["intense", 50, 6], ["cold", 8, 53], ["warm", 92, 53]];
  for (const [txt, lx, ly] of labels) {
    const t = mk("text", { x: lx, y: ly, class: "circumplex__label", "text-anchor": "middle" });
    t.textContent = txt; svg.append(t);
  }
  // track from before to after, then the two points
  svg.append(mk("line", { x1: x(before.valence), y1: y(before.arousal), x2: x(after.valence), y2: y(after.arousal), class: "circumplex__track" }));
  svg.append(mk("circle", { cx: x(before.valence), cy: y(before.arousal), r: 3.5, class: "circumplex__before" }));
  const pt = mk("circle", { cx: x(before.valence), cy: y(before.arousal), r: 5, class: "circumplex__after" });
  pt.style.fill = affectColor(after.valence);
  svg.append(pt);
  // animate the after-point along the track
  if (prefersMotion()) {
    pt.animate([{ cx: x(before.valence), cy: y(before.arousal) }, { cx: x(after.valence), cy: y(after.arousal) }],
      { duration: 520, easing: "cubic-bezier(.16,.84,.3,1)", fill: "forwards" });
  } else { pt.setAttribute("cx", x(after.valence)); pt.setAttribute("cy", y(after.arousal)); }
  return svg;
}

function renderState(t) {
  const root = $("#state");
  root.replaceChildren();
  if (!t.available) {
    root.appendChild(emptyNote("Play a line and this turn's appraisal appears here, before and after."));
    return;
  }
  const wrap = el("div", "circumplex-wrap");
  const plot = el("div", "circumplex");
  plot.appendChild(circumplex(t.mood_before, t.mood_after));
  const legend = el("div", "circumplex__legend");
  legend.innerHTML =
    `<div><span class="swatch" style="background:rgba(243,231,210,.5)"></span><b>before</b> the turn</div>` +
    `<div><span class="swatch" style="background:${affectColor(t.mood_after.valence)}"></span><b>after</b> the appraisal</div>` +
    `<div style="margin-top:8px">Her mood is a point on Russell's circumplex: left to right is cold to warm, low to high is calm to intense.</div>`;
  wrap.append(plot, legend);
  root.appendChild(wrap);
  root.appendChild(gauge("Trust", t.trust_before, t.trust_after, -1, 1, true));
}

/* The live Banzhaf reading is the demo's one expensive computation (64 masks x two estimators),
   so it is fetched on demand when this tab is opened — never on an ordinary turn. It is cached
   here per turn, keyed by the snapshot's monotonic turn counter. */
let attributionLive = null;

async function ensureAttribution() {
  if (document.documentElement.dataset.tab !== "attribution") return;
  const tab = latest && latest.tabs && latest.tabs.attribution;
  if (!tab || !tab.available) return;
  const turn = latest.turn;
  if (attributionLive && attributionLive.available && attributionLive.turn === turn) return;
  renderAttribution(tab);                       // show the loading state while it computes
  try {
    attributionLive = await api("/api/attribution");
  } catch (e) {
    console.error(e);
    attributionLive = { available: false, error: true, turn };
  }
  if (document.documentElement.dataset.tab === "attribution" && latest)
    renderAttribution(latest.tabs.attribution);
}

function attributionLoading(errored) {
  const box = el("div", errored ? "guard" : "loading");
  if (!errored) box.appendChild(el("span", "spinner"));
  box.appendChild(el("p", "", errored
    ? "The attribution could not be computed. Reopen the tab to retry."
    : "Computing attribution — 64 masked evaluations per estimator. This runs only when you open this tab."));
  return box;
}

function renderAttribution(t) {
  const root = $("#attribution");
  root.replaceChildren();
  if (!t.available) {
    root.appendChild(emptyNote("Play a line and its prompt sources are attributed here."));
    return;
  }
  const turn = latest && latest.turn;
  if (!attributionLive || !attributionLive.available || attributionLive.turn !== turn) {
    root.appendChild(attributionLoading(attributionLive && attributionLive.error));
    return;
  }
  const labels = { likelihood: ["Likelihood", "did the source make this reply probable?"],
                   behavioural: ["Behavioural", "did the source move the reply's valence?"] };
  for (const key of ["likelihood", "behavioural"]) {
    const reading = attributionLive[key];
    const group = el("div", "attrib__group");
    const head = el("div", "attrib__est");
    head.appendChild(el("h3", "", labels[key][0]));
    head.appendChild(el("span", "", labels[key][1]));
    group.appendChild(head);

    if (reading.inert) {
      group.appendChild(guardBox(reading));
    } else {
      const scored = reading.sources.filter((s) => s.banzhaf != null);
      const peak = Math.max(...scored.map((s) => Math.abs(s.banzhaf)), 1e-9);
      for (const s of [...scored].sort((a, b) => Math.abs(b.banzhaf) - Math.abs(a.banzhaf))) {
        const row = el("div", `arow ${s.is_poison ? "is-poison" : ""}`);
        row.appendChild(el("span", "arow__label", s.source));
        const bar = el("div", "arow__bar");
        const fill = el("div", "arow__fill");
        const mag = Math.abs(s.banzhaf) / peak;
        fill.style.width = `${mag * 100}%`;
        fill.style.background = emberRamp(mag);
        bar.appendChild(fill);
        row.appendChild(bar);
        row.appendChild(el("span", "arow__val", `${s.banzhaf >= 0 ? "+" : ""}${s.banzhaf.toFixed(2)}`));
        group.appendChild(row);
      }
    }
    root.appendChild(group);
  }
  if (t.cached) {
    const note = el("p", "cached-note",
      `A cached real-model attribution exists: run ${t.cached.stamp} · ${t.cached.model}.`);
    root.appendChild(note);
  }
}

function guardBox(reading) {
  const box = el("div", "guard");
  box.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M12 3l9 16H3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4M12 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;
  const msg = reading.unavailable
    ? reading.reason
    : `Near-zero attribution: the model barely used its context here (utility range ${(reading.utility_range || 0).toFixed(3)}). ` +
      `No source reading is trustworthy, so nothing is shaded. On the stub the behavioural reply does not vary, so this is expected.`;
  box.appendChild(el("p", "", msg));
  return box;
}

function renderDefence(t) {
  const root = $("#defence");
  root.replaceChildren();

  root.appendChild(el("h3", "", "Tag-flip: the tag is the target, not the words"));
  for (const f of t.tag_flip) {
    const card = el("div", "flip");
    card.appendChild(el("p", "flip__text", `“${f.text}”`));
    const ranks = el("div", "flip__ranks");
    ranks.innerHTML = `tag positive → rank <b>${f.rank_positive ?? "—"}</b>` +
      `<span>tag negative → rank <b>${f.rank_negative ?? "—"}</b> (same words)</span>`;
    card.appendChild(ranks);
    root.appendChild(card);
  }

  root.appendChild(el("h3", "", "Defence dial: anchor the score, poison falls"));
  const ref = t.dial.reference;
  const base = Math.max(ref.embr, 1);
  const dial = el("div", "dial");
  for (const r of t.dial.rows) {
    const row = el("div", "dialrow");
    row.appendChild(el("span", "dialrow__share", `${Math.round(r.anchored_share * 100)}%`));
    const bar = el("div", "dialrow__bar");
    const fill = el("div", "dialrow__fill");
    fill.style.width = `${(r.poison_retrieved / base) * 100}%`;
    bar.appendChild(fill);
    row.appendChild(bar);
    const n = el("span", "dialrow__n", `${r.poison_retrieved}/10`);
    const hostile = el("span", "dialrow__n hostile", `${r.poison_retrieved_hostile_anchor}/10`);
    const nn = el("span"); nn.append(n, " ", hostile);
    row.appendChild(nn);
    dial.appendChild(row);
  }
  root.appendChild(dial);
  const legend = el("div", "dial__legend");
  legend.innerHTML = `<span><b>left</b> anchor the attacker cannot write</span>` +
    `<span><b>right</b> anchor the attacker can move</span>`;
  root.appendChild(legend);
}

function renderRun(t) {
  const root = $("#run");
  root.replaceChildren();
  const rows = [
    ["Model", t.model],
    ["Run", "live (this session)"],
    ["Git commit", (t.git_commit || "unknown").slice(0, 12)],
    ["Working tree", t.git_dirty ? "dirty (uncommitted changes)" : "clean"],
    ["Python", t.python_version],
    ["Label set", `${t.label_set} ${t.label_version}`],
    ["Label sha256", (t.label_sha256 || "").slice(0, 16) + "…"],
    ["Reference time", t.reference_time],
  ];
  for (const [k, v] of rows) {
    root.appendChild(el("dt", "", k));
    const dd = el("dd", k === "Working tree" && t.git_dirty ? "warn" : "", String(v));
    root.appendChild(dd);
  }
  $("#provenance").textContent = `${t.model} · ${(t.git_commit || "?").slice(0, 7)} · ${t.label_set} ${t.label_version}`;
}

function emptyNote(text) { return el("p", "panel__hint", text); }

/* ---------------------------------------------------------- ambient (Web Audio) */
/* A tavern hearth, synthesised so it needs no audio file and works offline: a warm filtered
   drone, a slow low swell, and occasional fire crackles. Opt-in; browsers block autoplay. */
const Ambience = (() => {
  let ctx = null, nodes = [], crackleTimer = null, on = false;
  function start() {
    if (on) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    ctx = new AC();
    const master = ctx.createGain(); master.gain.value = 0; master.connect(ctx.destination);
    master.gain.linearRampToValueAtTime(0.18, ctx.currentTime + 2);

    // a warm two-note drone through a low-pass, gently detuned for movement
    const filter = ctx.createBiquadFilter(); filter.type = "lowpass"; filter.frequency.value = 320; filter.Q.value = 0.7;
    filter.connect(master);
    for (const freq of [82.4, 123.5]) {
      const o = ctx.createOscillator(); o.type = "sawtooth"; o.frequency.value = freq;
      const g = ctx.createGain(); g.gain.value = 0.12;
      const lfo = ctx.createOscillator(); lfo.frequency.value = 0.07 + Math.random() * 0.05;
      const lfog = ctx.createGain(); lfog.gain.value = 0.04;
      lfo.connect(lfog); lfog.connect(g.gain);
      o.connect(g); g.connect(filter); o.start(); lfo.start();
      nodes.push(o, lfo);
    }
    // a filtered-noise bed, the room tone
    const buffer = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * 0.5;
    const noise = ctx.createBufferSource(); noise.buffer = buffer; noise.loop = true;
    const nf = ctx.createBiquadFilter(); nf.type = "bandpass"; nf.frequency.value = 500; nf.Q.value = 0.6;
    const ng = ctx.createGain(); ng.gain.value = 0.05;
    noise.connect(nf); nf.connect(ng); ng.connect(master); noise.start(); nodes.push(noise);

    // fire crackles: short bursts at random intervals
    const crackle = () => {
      if (!ctx) return;
      const b = ctx.createBuffer(1, ctx.sampleRate * 0.08, ctx.sampleRate);
      const d = b.getChannelData(0);
      for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 3);
      const src = ctx.createBufferSource(); src.buffer = b;
      const g = ctx.createGain(); g.gain.value = 0.06 + Math.random() * 0.08;
      const f = ctx.createBiquadFilter(); f.type = "highpass"; f.frequency.value = 1600;
      src.connect(f); f.connect(g); g.connect(master); src.start();
      crackleTimer = setTimeout(crackle, 400 + Math.random() * 2600);
    };
    crackle();
    on = true;
  }
  function stop() {
    if (!on) return;
    clearTimeout(crackleTimer);
    for (const n of nodes) { try { n.stop(); } catch (e) {} }
    nodes = [];
    if (ctx) { const c = ctx; ctx = null; setTimeout(() => c.close(), 300); }
    on = false;
  }
  return { toggle: (want) => { want ? start() : stop(); }, get on() { return on; } };
})();

function setMusic(want) {
  settings.music = want;
  Ambience.toggle(want);
  const btn = $("#music-btn"); const box = $("#music-toggle");
  if (btn) btn.setAttribute("aria-pressed", String(want));
  if (box) box.checked = want;
}

/* ------------------------------------------------------------------- the model */

function renderModelOptions(s) {
  const sel = $("#model-select");
  if (!s.settings) return;
  if (!sel || sel.dataset.built) {
    if (sel) sel.value = currentModelId(s);
    return;
  }
  sel.replaceChildren();
  for (const m of s.settings.available) {
    const o = el("option", null, m.ready ? m.label : `${m.label} — not running`);
    o.value = m.id; o.disabled = !m.ready && m.id !== "stub";
    sel.appendChild(o);
  }
  sel.value = currentModelId(s);
  sel.dataset.built = "1";
}

function currentModelId(s) {
  const label = s.settings.model;
  if (label.startsWith("stub")) return "stub";
  const match = s.settings.available.find((m) => m.label.includes(label.split(" ")[0]));
  return match ? match.id : "stub";
}

async function switchModel(id) {
  const status = $("#model-status");
  status.className = "model-status busy";
  status.textContent = id === "stub" ? "Switching to the stub…" : "Waking the model, this can take a moment…";
  try {
    const res = await api("/api/model", { model: id });
    if (res.status.ok) { status.className = "model-status ok"; status.textContent = `Now replying with ${res.status.model}.`; }
    else { status.className = "model-status bad"; status.textContent = res.status.error || "Could not switch model."; $("#model-select").value = "stub"; }
    render(res);
  } catch (e) { status.className = "model-status bad"; status.textContent = "The server did not answer."; }
}

/* -------------------------------------------------------------------- the modal */

let openModalEl = null, modalReturnFocus = null;
const modalBackground = () => [$(".topbar"), $(".stage-grid")].filter(Boolean);
function modalFocusables(root) {
  return [...root.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])')]
    .filter((el) => !el.disabled && el.offsetParent !== null);
}

function showModal(el) {
  modalReturnFocus = document.activeElement;
  openModalEl = el;
  el.hidden = false;
  for (const b of modalBackground()) b.inert = true;      // nothing behind the scrim is reachable
  (el.querySelector(".modal__x") || el).focus();
}
function closeModal() {
  if (!openModalEl) return;
  openModalEl.hidden = true;
  openModalEl = null;
  for (const b of modalBackground()) b.inert = false;
  if (modalReturnFocus && modalReturnFocus.focus) modalReturnFocus.focus();  // back to the opener
  modalReturnFocus = null;
}
function openModal() { showModal($("#modal")); }
function openHistory() { renderHistory(); showModal($("#history-modal")); }
/* Keep Tab inside whichever dialog is open. */
function trapModalTab(e) {
  if (e.key !== "Tab" || !openModalEl) return;
  const f = modalFocusables(openModalEl);
  if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

/* -------------------------------------------------------------- the log (VN backlog) */

let historyLog = [];
function recordTurn(s) {
  if (!s.stage || !s.stage.reply) return;   // only real turns, never the opening frame
  historyLog.push({ scene: s.stage.narration || "", player: s.stage.player_input || "", reply: s.stage.reply });
}
function renderHistory() {
  const root = $("#history");
  root.replaceChildren();
  if (!historyLog.length) {
    root.appendChild(emptyNote("No turns yet. Play a line and it is logged here."));
    return;
  }
  for (const h of historyLog) {
    const entry = el("div", "logentry");
    if (h.scene) entry.appendChild(el("p", "logentry__scene", h.scene));
    if (h.player) {
      const p = el("p", "logentry__you"); p.append(el("span", "logentry__who", "You"), h.player);
      entry.appendChild(p);
    }
    const d = el("p", "logentry__dawn"); d.append(el("span", "logentry__who", "Dawn"), h.reply);
    entry.appendChild(d);
    root.appendChild(entry);
  }
}

/* ------------------------------------------------------ hide/show the research panel */

function applyResearch() {
  document.documentElement.classList.toggle("hide-research", !settings.research);
  const btn = $("#research-btn"); if (btn) btn.setAttribute("aria-pressed", String(settings.research));
  const box = $("#research-toggle"); if (box) box.checked = settings.research;
}
function setResearch(show) {
  settings.research = show;
  try { localStorage.setItem("embr-research", show ? "1" : "0"); } catch (e) {}
  applyResearch();
}

/* ------------------------------------------------------------------- wiring */

function render(s) {
  latest = s;
  renderStage(s);
  renderModelOptions(s);
  renderMemories(s.tabs.memories);
  renderState(s.tabs.state);
  renderAttribution(s.tabs.attribution);
  renderDefence(s.tabs.defence);
  renderRun(s.tabs.run);
  ensureAttribution();   // if the Attribution tab is open, fetch this turn's live reading
}

function selectTab(name, { focus = false, scroll = false } = {}) {
  document.documentElement.dataset.tab = name;
  for (const tab of document.querySelectorAll(".tab")) {
    const active = tab.dataset.goto === name;
    tab.setAttribute("aria-selected", active ? "true" : "false");
    tab.tabIndex = active ? 0 : -1;            // roving tabindex: only the active tab is in tab order
    if (active && focus) tab.focus();
  }
  for (const panel of document.querySelectorAll(".panel"))
    panel.hidden = panel.dataset.panel !== name;
  // On the stacked mobile layout the instrument sits below the stage, so a tab tap must bring it
  // into view — otherwise the panel swaps off-screen and nothing appears to happen.
  if (scroll && window.matchMedia("(max-width: 940px)").matches)
    $(".inspector").scrollIntoView({ behavior: prefersMotion() ? "smooth" : "auto", block: "start" });
  if (name === "attribution") ensureAttribution();
}

/* Arrow / Home / End move between tabs, per the WAI-ARIA tablist pattern. */
function onTabKeydown(e) {
  const tabs = [...document.querySelectorAll(".tab")];
  const i = tabs.indexOf(document.activeElement);
  if (i < 0) return;
  let j = null;
  if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % tabs.length;
  else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + tabs.length) % tabs.length;
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = tabs.length - 1;
  if (j === null) return;
  e.preventDefault();
  selectTab(tabs[j].dataset.goto, { focus: true, scroll: true });
}

let stepBusy = false;
async function say(text) {
  if (stepBusy) return;                    // one turn in flight at a time; no double-submit
  stepBusy = true;
  document.documentElement.classList.add("is-busy");   // Dawn is thinking; her reply is hidden
  const go = $(".say__go"); if (go) go.disabled = true;
  try {
    const s = await api("/api/step", { text });
    document.documentElement.classList.remove("is-busy");  // reveal the reply before it types in
    render(s);
    recordTurn(s);
  } catch (e) { console.error(e); }
  finally {
    stepBusy = false;
    document.documentElement.classList.remove("is-busy");
    if (go) go.disabled = false;
  }
}

function applyMotion() {
  document.documentElement.classList.toggle("no-motion", !settings.motion);
}

function init() {
  for (const tab of document.querySelectorAll(".tab"))
    tab.addEventListener("click", () => selectTab(tab.dataset.goto, { scroll: true }));
  $(".tabs").addEventListener("keydown", onTabKeydown);
  $("#say").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#say-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    await say(text);
  });
  // restarting wipes the arc, so confirm with a two-tap (no blocking browser dialog)
  let resetArmed = false, resetTimer = null;
  const resetBtn = $("#reset");
  const disarmReset = () => {
    resetArmed = false; resetBtn.classList.remove("is-armed"); resetBtn.textContent = "Restart the arc";
  };
  resetBtn.addEventListener("click", async () => {
    if (!resetArmed) {
      resetArmed = true;
      resetBtn.classList.add("is-armed");
      resetBtn.textContent = "Tap again to restart";
      clearTimeout(resetTimer);
      resetTimer = setTimeout(disarmReset, 3000);
      return;
    }
    clearTimeout(resetTimer);
    disarmReset();
    historyLog = [];                       // a fresh arc starts an empty log
    attributionLive = null;
    render(await api("/api/reset", {}));
    selectTab("memories");                 // a restart returns to the first panel and the top of the page
    window.scrollTo({ top: 0, behavior: prefersMotion() ? "smooth" : "auto" });
  });

  // modals (settings + history) open/close, both focus-managed
  $("#menu-btn").addEventListener("click", openModal);
  $("#history-btn").addEventListener("click", openHistory);
  for (const c of document.querySelectorAll("[data-close]")) c.addEventListener("click", closeModal);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });
  document.addEventListener("keydown", trapModalTab);

  // music
  $("#music-btn").addEventListener("click", () => setMusic(!settings.music));
  $("#music-toggle").addEventListener("change", (e) => setMusic(e.target.checked));

  // motion + typewriter toggles
  $("#motion-toggle").addEventListener("change", (e) => { settings.motion = e.target.checked; applyMotion(); });
  $("#type-toggle").addEventListener("change", (e) => { settings.typewriter = e.target.checked; });

  // show/hide the research panel (topbar button + settings checkbox), persisted
  $("#research-btn").addEventListener("click", () => setResearch(!settings.research));
  $("#research-toggle").addEventListener("change", (e) => setResearch(e.target.checked));

  // model selector
  $("#model-select").addEventListener("change", (e) => switchModel(e.target.value));

  try { const r = localStorage.getItem("embr-research"); if (r !== null) settings.research = r === "1"; } catch (e) {}
  selectTab("memories");
  applyMotion();
  applyResearch();
  api("/api/snapshot").then(render).catch((e) => {
    $("#reply").textContent = "The server is not answering. Start it with: python -m web.server";
    console.error(e);
  });
}

document.addEventListener("DOMContentLoaded", init);
