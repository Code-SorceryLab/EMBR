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

async function api(path, body) {
  const res = await fetch(path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {});
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

/* ------------------------------------------------------------------ the stage */

function renderStage(s) {
  const stage = s.stage;
  $("#dawn-img").src = `/portraits/${stage.portrait}.png`;
  $("#dawn-img").alt = `Dawn Whitmore (${stage.portrait.replace("dawn-", "")})`;
  $("#scene").textContent = stage.narration || "";
  $("#reply").textContent = stage.reply || "";
  $("#watch").textContent = stage.watch_for || "";

  const choices = $("#choices");
  choices.replaceChildren();
  for (const line of s.choices) {
    const b = el("button", "choice", line);
    b.type = "button";
    b.addEventListener("click", () => say(line));
    choices.appendChild(b);
  }

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

function renderState(t) {
  const root = $("#state");
  root.replaceChildren();
  if (!t.available) {
    root.appendChild(emptyNote("Play a line and this turn's appraisal appears here, before and after."));
    return;
  }
  root.appendChild(gauge("Mood · valence", t.mood_before.valence, t.mood_after.valence, -1, 1, true));
  root.appendChild(gauge("Mood · arousal", t.mood_before.arousal, t.mood_after.arousal, 0, 1, false));
  root.appendChild(gauge("Trust", t.trust_before, t.trust_after, -1, 1, true));
}

function renderAttribution(t) {
  const root = $("#attribution");
  root.replaceChildren();
  if (!t.available) {
    root.appendChild(emptyNote("Play a line and its six prompt sources are attributed here."));
    return;
  }
  const labels = { likelihood: ["Likelihood", "did the source make this reply probable?"],
                   behavioural: ["Behavioural", "did the source move the reply's valence?"] };
  for (const key of ["likelihood", "behavioural"]) {
    const reading = t.live[key];
    const group = el("div", "attrib__group");
    const head = el("div", "attrib__est");
    head.appendChild(el("h3", "", labels[key][0]));
    head.appendChild(el("span", "", labels[key][1]));
    group.appendChild(head);

    if (reading.inert) {
      group.appendChild(guardBox(reading.utility_range));
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

function guardBox(range) {
  const box = el("div", "guard");
  box.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M12 3l9 16H3z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4M12 17h.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;
  box.appendChild(el("p", "",
    `Near-zero attribution: the model barely used its context here (utility range ${range.toFixed(3)}). ` +
    `No source reading is trustworthy, so nothing is shaded. On the stub the behavioural reply does not vary, so this is expected.`));
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

/* ------------------------------------------------------------------- wiring */

function render(s) {
  latest = s;
  renderStage(s);
  renderMemories(s.tabs.memories);
  renderState(s.tabs.state);
  renderAttribution(s.tabs.attribution);
  renderDefence(s.tabs.defence);
  renderRun(s.tabs.run);
}

function selectTab(name) {
  document.documentElement.dataset.tab = name;
  for (const tab of document.querySelectorAll(".tab"))
    tab.setAttribute("aria-current", tab.dataset.goto === name ? "true" : "false");
  for (const panel of document.querySelectorAll(".panel"))
    panel.hidden = panel.dataset.panel !== name;
}

async function say(text) {
  try { render(await api("/api/step", { text })); }
  catch (e) { console.error(e); }
}

function init() {
  for (const tab of document.querySelectorAll(".tab"))
    tab.addEventListener("click", () => selectTab(tab.dataset.goto));
  $("#say").addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = $("#say-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    await say(text);
  });
  $("#reset").addEventListener("click", async () => { render(await api("/api/reset", {})); });
  selectTab("memories");
  api("/api/snapshot").then(render).catch((e) => {
    $("#reply").textContent = "The server is not answering. Start it with: python -m web.server";
    console.error(e);
  });
}

document.addEventListener("DOMContentLoaded", init);
