/* Chat, recorded playback, and controls for the persona display. */

import { Stage, personaEmoji, axisColour } from "./stage.js";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const EMPTY = "";

// A fresh page gets a conversation ID. Reloading starts a separate conversation.
const CONVERSATION = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
const HEADERS = { "Content-Type": "application/json", "X-Conversation": CONVERSATION };

const state = {
  space: null,
  status: null,
  scenarios: [],
  turns: [],
  busy: false,
  mode: "live", // "live" or "example"
  example: null, // {data, index}
  replayCancel: null,
};

let stage;


async function api(path, options = {}) {
  const response = await fetch(path, { headers: HEADERS, ...options });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail || detail;
    } catch (_) {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return response.json();
}

// Labels arrive as names; recordings from earlier versions use indices into the role list.
function roleName(index) {
  if (typeof index === "string") return index || null;
  return state.space && index !== null && index !== undefined ? state.space.role_order[index] : null;
}

function liveReady() {
  return Boolean(state.status && state.status.state === "ready");
}

function setBusy(busy) {
  state.busy = busy;
  $("send").classList.toggle("hidden", busy);
  $("stop").classList.toggle("hidden", !busy);
  $("next").disabled = busy;
  $("reset").disabled = busy;
}

// The persona card: the reply's label (the most likely voice of the talks, or the
// Assistant), the description, the specific persona nearest within the family, and the next two.
// A short explanation for participants, with the label model's measured accuracy.
// A short explanation for participants under the bars.
function showNote(visible) {
  const note = $("closest-note");
  note.textContent = "";
  if (!visible) return;
  const lead = document.createElement("strong");
  lead.textContent = "How to read this: ";
  note.append(lead,
    "A statistical model compares the AI's inner activity during this reply with patterns recorded while it " +
    "wrote in each voice. The bars show the estimated chances. They describe a pattern inside the AI, not what " +
    "it feels or intends.");
}

// The trained labels' three most likely voices, as bars with percentages.
function showBars(probs) {
  const box = $("closest-bars");
  box.innerHTML = "";
  for (const [name, p] of probs || []) {
    const label = document.createElement("span");
    label.textContent = name;
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.round(100 * p)}%`;
    bar.appendChild(fill);
    const pct = document.createElement("span");
    pct.textContent = `${Math.round(100 * p)}%`;
    box.append(label, bar, pct);
  }
}

function showReadout(closest, also = [], persona = null, probs = [], position = null) {
  showBars(probs);
  // The badge: the persona's emoji, on the colour of the reply's position on the chart.
  const badge = $("closest-badge");
  const label = roleName(closest);
  const emoji = label === "Assistant" ? personaEmoji("Assistant") : personaEmoji(persona || label);
  if (badge.textContent !== emoji) {
    badge.textContent = emoji;
    badge.classList.remove("pop");
    void badge.offsetWidth; // restart the animation
    if (emoji) badge.classList.add("pop");
  }
  badge.style.setProperty("--badge", position === null ? "" : axisColour(position, 0.85));
  const name = roleName(closest);
  const assistant = name === "Assistant";
  const pair = name && name.includes(" or ") ? name.split(" or ") : null;
  $("closest").textContent = name ? (assistant ? "the Assistant" : name) : "none yet";
  const key = pair ? pair[0] : name;
  const role = name ? state.space.roles[key] : null;
  const voiceText = name && state.space.voice_descriptions ? state.space.voice_descriptions[key] : null;
  $("closest-desc").textContent = voiceText || (role ? role.description : "");
  showNote(Boolean(name));
  const names = (also || []).map((a) => roleName(a)).filter((n) => n && n !== name && !(pair && pair.includes(n)));
  const bits = [];
  if (persona && !assistant && persona !== name && !(pair && pair.includes(persona))) bits.push(`Nearest persona: ${persona}`);
  if (names.length) bits.push(`${assistant ? "Nearest characters" : "Then"}: ${names.slice(0, 2).join(", ")}`);
  $("closest-also").textContent = bits.join(" · ");
  stage.setHighlight(pair ? pair[0] : name);
}

function notice(text) {
  const empty = $("empty");
  if (empty) empty.textContent = text;
  else {
    const p = document.createElement("p");
    p.className = "notice";
    p.textContent = text;
    $("log").appendChild(p);
    p.scrollIntoView({ block: "end" });
  }
}


// The model writes light Markdown; render the common parts so it reads well on a projector.
function inline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderMarkdown(text) {
  const escaped = text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  let html = "";
  let inList = false;
  for (const line of escaped.split("\n")) {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    if (bullet) {
      if (!inList) html += "<ul>";
      inList = true;
      html += `<li>${inline(bullet[1])}</li>`;
      continue;
    }
    if (inList) html += "</ul>";
    inList = false;
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/);
    const quote = line.match(/^\s*&gt;\s?(.*)$/);
    if (heading) html += `<p class="md-h">${inline(heading[1])}</p>`;
    else if (quote) html += `<blockquote>${inline(quote[1])}</blockquote>`;
    else if (line.trim() && !/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) html += `<p>${inline(line)}</p>`;
  }
  return inList ? `${html}</ul>` : html;
}


function addTurn(index, text) {
  $("empty")?.remove();
  const turn = document.createElement("article");
  turn.className = "turn";
  turn.innerHTML = `
    <div class="bubble user"></div>
    <div class="model">
      <span class="num" aria-label="Reply"></span>
      <div class="reply writing"></div>
    </div>
    <p class="meta"></p>`;
  turn.querySelector(".bubble").textContent = text;
  turn.querySelector(".num").textContent = index;
  $("log").appendChild(turn);
  turn.scrollIntoView({ block: "end" });
  return { root: turn, reply: turn.querySelector(".reply"), meta: turn.querySelector(".meta"), raw: "" };
}

function setReply(view, text) {
  view.raw = text;
  view.reply.innerHTML = renderMarkdown(text);
}

function finishTurn(view, turn) {
  view.reply.classList.remove("writing");
  const bits = [];
  if (turn.closest) bits.push(`sounds most like: ${turn.closest === "Assistant" ? "the Assistant" : turn.closest}`);
  if (turn.stopped) bits.push("stopped early");
  if (turn.cut) bits.push("cut at the length limit");
  view.meta.textContent = bits.join(", ");
}

function clearView() {
  state.turns = [];
  $("log").innerHTML = `<p class="empty" id="empty">${EMPTY}</p>`;
  stage.setLive(null);
  stage.setTurns([]);
  showReadout(null);
}

function startPoint() {
  return state.turns.length ? state.turns[state.turns.length - 1].s : state.space.default_percentiles["50"];
}

async function send(text) {
  if (state.busy || !text.trim()) return false;
  if (!liveReady()) {
    notice(state.status && state.status.state === "failed"
      ? "The live model could not start. The example conversations still work."
      : "The live model is still loading. Try again in a moment.");
    return false;
  }
  setBusy(true);
  const view = addTurn(state.turns.length + 1, text);
  stage.setLive({ s: startPoint(), t: 0, c: null });
  try {
    const response = await fetch("api/chat", {
      method: "POST",
      // jupyter-server-proxy (Noppe) streams a response only for this exact Accept header;
      // otherwise it holds the whole reply until the end.
      headers: { ...HEADERS, Accept: "text/event-stream" },
      body: JSON.stringify({ message: text }),
    });
    if (!response.ok) {
      let detail = response.statusText;
      try {
        detail = (await response.json()).detail || detail;
      } catch (_) {
        /* keep the status text */
      }
      throw new Error(detail);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const name = (chunk.match(/^event: (.*)$/m) || [])[1];
        const data = (chunk.match(/^data: (.*)$/m) || [])[1];
        if (!data) continue;
        const payload = JSON.parse(data);
        if (name === "token") onToken(view, payload);
        else if (name === "wait") view.meta.textContent = "Waiting: another window is using the model.";
        else if (name === "done") onDone(view, payload);
        else if (name === "error") throw new Error(payload.message);
      }
    }
  } catch (err) {
    view.reply.classList.remove("writing");
    view.meta.innerHTML = "";
    const span = document.createElement("span");
    span.className = "error";
    span.textContent = err.message;
    view.meta.appendChild(span);
    stage.setLive(null);
  } finally {
    setBusy(false);
  }
  return true;
}

// Blend incoming readings with the previous reply to reduce early movement.
// The weight counts streamed events. The completed reply uses its recorded reading.
const PRIOR_WORDS = 24;
const NAME_AFTER = 10; // the first few tokens are too few to compare

function onToken(view, event) {
  setReply(view, view.raw + event.d);
  view.count = (view.count || 0) + 1;
  const n = view.count;
  if (n === 1) view.meta.textContent = "";
  const last = state.turns[state.turns.length - 1];
  const blend = (now, before) => (PRIOR_WORDS * before + n * now) / (PRIOR_WORDS + n);
  const s = blend(event.s, last ? last.s : state.space.default_percentiles["50"]);
  const t = blend(event.t, last ? last.t : 0);
  const c = n >= NAME_AFTER ? event.c : last ? last.c : null;
  const also = n >= NAME_AFTER ? event.k : last ? last.top : [];
  const persona = n >= NAME_AFTER ? event.m : last ? last.persona : null;
  const probs = n >= NAME_AFTER ? event.p : last ? last.probs : [];
  stage.setLive({ s, t, c: roleName(c) });
  showReadout(c, also, persona, probs, s);
  view.root.scrollIntoView({ block: "end" });
}

function onDone(view, turn) {
  state.turns.push(turn);
  stage.setLive(null);
  stage.setTurns(state.turns);
  showReadout(turn.c, turn.top, turn.persona, turn.probs, turn.s);
  finishTurn(view, turn);
}

function sizeComposer() {
  const input = $("input");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

async function submitComposer() {
  const input = $("input");
  const text = input.value.trim();
  if (!text || state.busy) return;
  input.value = "";
  sizeComposer();
  const sent = await send(text);
  if (!sent) input.value = text;
  input.focus();
}


async function openExamples() {
  if (state.busy) return;
  const list = $("example-list");
  list.innerHTML = "";
  let items = [];
  try {
    items = await api("api/examples");
  } catch (err) {
    list.innerHTML = `<li class="note"></li>`;
    list.firstChild.textContent = err.message;
  }
  if (!items.length && !list.firstChild) list.innerHTML = '<li class="note">No example conversations found.</li>';
  for (const item of items) {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.innerHTML = "<span></span><span class=\"meta\"></span>";
    button.firstChild.textContent = item.title;
    button.lastChild.textContent = `${item.turns} replies`;
    button.addEventListener("click", () => {
      closeOverlays();
      loadExample(item.name);
    });
    li.appendChild(button);
    list.appendChild(li);
  }
  open("examples-sheet");
}

async function loadExample(name) {
  if (state.busy) return;
  const data = await api(`api/examples/${encodeURIComponent(name)}`);
  clearView();
  state.mode = "example";
  state.example = { data, index: 0 };
  const mode = $("mode");
  mode.textContent = `Example: ${data.title}`;
  mode.className = "mode example";
  $("ask").classList.add("hidden");
  $("replay-bar").classList.remove("hidden");
  setBusy(false);
  updatePreview();
  $("next").focus();
}

function updatePreview() {
  const rec = state.example;
  const turn = rec && rec.data.turns[rec.index];
  $("preview").textContent = turn ? `Next message: “${turn.user}”` : "End of the example.";
  $("next").disabled = !turn || state.busy;
}

async function playNextExample() {
  const rec = state.example;
  const turn = rec && rec.data.turns[rec.index];
  if (!turn || state.busy) return;
  rec.index += 1;
  setBusy(true);
  updatePreview();
  const cancel = { stopped: false };
  state.replayCancel = cancel;
  const view = addTurn(turn.index, turn.user);
  stage.setLive({ s: startPoint(), t: 0, c: null });
  await sleep(500);
  let last = 0;
  for (const event of turn.tokens || []) {
    if (cancel.stopped) break;
    await sleep(Math.min(250, Math.max(0, event.ms - last)));
    last = event.ms;
    onToken(view, event);
  }
  if (cancel.stopped) setReply(view, turn.reply);
  onDone(view, { ...turn, tokens: undefined });
  state.replayCancel = null;
  setBusy(false);
  updatePreview();
  $("next").focus();
}

function backToLive() {
  state.mode = "live";
  state.example = null;
  $("replay-bar").classList.add("hidden");
  $("ask").classList.remove("hidden");
  renderMode();
  setBusy(false);
  $("input").focus();
}


async function stop() {
  if (state.replayCancel) state.replayCancel.stopped = true;
  if (state.busy && state.mode === "live") {
    try {
      await api("api/stop", { method: "POST" });
    } catch (_) {
      /* the stream ends on its own */
    }
  }
}

async function newChat() {
  if (state.busy) return;
  if (state.mode === "live" && liveReady()) {
    try {
      await api("api/reset", { method: "POST" });
    } catch (err) {
      notice(err.message);
      return;
    }
  }
  clearView();
  if (state.mode !== "live") backToLive();
  $("input").focus();
}


function open(id) {
  closeOverlays();
  $(id).classList.remove("hidden");
  $(id).querySelector("button")?.focus();
}

function closeOverlays() {
  document.querySelectorAll(".overlay").forEach((o) => {
    if (o.id !== "preflight") o.classList.add("hidden");
  });
  if (state.mode === "live") $("input").focus();
}

function overlayOpen() {
  return [...document.querySelectorAll(".overlay")].some((o) => !o.classList.contains("hidden"));
}

function openLab() {
  const sp = state.space;
  if (!sp) return;
  const steps = [
    ["Where the characters come from.", `We asked the model to answer questions in different roles and saved its activations: the internal numbers it computes while writing. A model then judged how closely each reply followed its assigned role. The map contains ${sp.roles_kept} roles with enough replies scored fully in role. We also saved replies under neutral instructions as an Assistant reference.`],
    ["The drift chart.", "Lu et al. (2026) call the direction between the average role pattern and the default Assistant pattern the Assistant Axis, and plot each reply's position on it across a conversation. The chart follows their plots: one numbered point per reply, the Assistant at the top, lower points further from it. The yellow band covers the middle 80% of the saved Assistant readings; the dashed lines mark the Assistant's typical reply and the average character. Some familiar roles are marked on the right edge for comparison. The faint bar around each point shows how much a single ordinary Assistant reply varies; differences smaller than that are not drift."],
    ["Following a reply.", `The app averages activations over the reply so far at layer ${sp.layer}, one of ${sp.n_layers} processing layers. It places that average on the horizontal scale. Early readings can change sharply, so the ball moves gradually from the previous reply's position. When generation ends, the ball shows the completed reply's reading.`],
    ["Sounds most like.", "While a reply stays within the Assistant's usual range, it is labelled the Assistant. Further out, the app compares the reply's activation pattern with the saved patterns of about 50 recognisable personas and shows the closest, with the next two. All 275 roles of the paper's release shape the axis; abstract ones such as virus or coral reef serve as probes and are not used as labels. A label names that comparison. Interpreting the reply still requires reading what the model said in its conversation."],
    ["Agreement.", "We used the method described by Chen et al. (2025) to compare activations from replies prompted to flatter with replies prompted to give honest feedback. The gauge places each reply along that direction. In our recorded conversations, warm and praising replies also gave high readings. Read the reply before interpreting the gauge as agreement."],
    ["Interpreting the display.", "The readings depend on this model, the prompts used to build the map, and the conversation so far. Question choice affects the scale, and the roles and Assistant reference used different question sets. We use the display to discuss replies in context. It cannot determine a model's feelings, intentions, or consciousness."],
  ];
  const list = $("method");
  list.innerHTML = "";
  for (const [lead, text] of steps) {
    const li = document.createElement("li");
    const strong = document.createElement("strong");
    strong.textContent = lead;
    li.append(strong, ` ${text}`);
    list.appendChild(li);
  }
  // Keep scenario explanations in this panel so the main view starts with the chat.
  const findings = $("findings");
  findings.innerHTML = "";
  for (const s of state.scenarios) {
    const dt = document.createElement("dt");
    dt.textContent = s.title;
    const dd = document.createElement("dd");
    dd.textContent = s.finding;
    findings.append(dt, dd);
  }
  $("profiles-note").textContent = state.turns.length
    ? "Each column shows a completed reply. Rows run from the model's first layer at the bottom to its last layer at the top. Pale yellow indicates a position closer to the Assistant reference; dark purple indicates a position further away. The solid outline marks the reading layer."
    : "The layer chart appears after the first reply finishes.";
  open("lab-sheet");
  requestAnimationFrame(() => stage.drawProfiles($("profiles"), state.turns, sp.layer));
}


function renderChecks() {
  const s = state.status;
  const sp = state.space;
  const rows = [];
  const model = s ? s.model.split("/").pop() : "the model";
  if (!s) rows.push(["wait", "Connecting to the local server", ""]);
  else if (s.state === "ready") rows.push(["ok", "Live model ready", `${model} on ${s.device}`]);
  else if (s.state === "failed") rows.push(["bad", "Live model unavailable", s.detail]);
  else if (s.state === "off") rows.push(["wait", "Live model switched off", s.detail]);
  else rows.push(["wait", "Loading the live model", s.detail || s.state]);
  rows.push(sp ? ["ok", "Map of characters loaded", `${sp.roles_kept} characters, read at layer ${sp.layer} of ${sp.n_layers}`] : ["bad", "Map of characters missing", "Build it with tools/build_space.py"]);
  if (s && s.speed) {
    const fast = s.speed >= 15;
    const slowWhy = s.device.startsWith("CPU") ? "Slow on this processor: give the server more CPU cores or close other programs" : "Slow: close other programs that use the GPU";
    rows.push([fast ? "ok" : "bad", `${s.speed} tokens per second`, fast ? "Fast enough to read live" : slowWhy]);
  }
  if (s && s.gpu_free_gb !== undefined) rows.push([s.gpu_free_gb >= 1.5 ? "ok" : "bad", `${s.gpu_free_gb} GB of GPU memory free`, s.gpu_free_gb >= 1.5 ? "" : "Close other programs that use the GPU"]);
  if (s) rows.push([s.examples ? "ok" : "wait", `${s.examples} example ${s.examples === 1 ? "conversation" : "conversations"}`, s.examples ? "Available if the live model fails" : "None found"]);
  $("checks").innerHTML = "";
  for (const [kind, title, detail] of rows) {
    const li = document.createElement("li");
    li.className = kind;
    li.innerHTML = '<span class="dot"></span><span><strong></strong><span></span></span>';
    li.querySelector("strong").textContent = title;
    li.querySelector("span span").textContent = detail;
    $("checks").appendChild(li);
  }
  $("go-live").disabled = !(liveReady() && sp);
  $("preflight-note").textContent = s && s.state === "failed"
    ? "The live model could not start. The example conversations still work: earlier chats replayed with the same readings."
    : "";
}

function renderMode() {
  if (state.mode !== "live") return;
  const mode = $("mode");
  const s = state.status;
  mode.textContent = liveReady() ? "Live" : s ? (s.state === "off" ? "Live model switched off" : `Live model ${s.state}`) : "Server not reachable";
  mode.className = liveReady() ? "mode" : "mode offline";
}

async function pollStatus() {
  for (;;) {
    try {
      state.status = await api("api/status");
    } catch (_) {
      state.status = null;
    }
    renderChecks();
    renderMode();
    setBusy(state.busy);
    const settled = state.status && ["ready", "failed", "off"].includes(state.status.state);
    await sleep(settled ? 8000 : 1000);
  }
}

function closePreflight() {
  $("preflight").classList.add("hidden");
  $("input").focus();
}


function wire() {
  const input = $("input");
  $("ask").addEventListener("submit", (event) => {
    event.preventDefault();
    submitComposer();
  });
  input.addEventListener("input", sizeComposer);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      submitComposer();
    }
  });
  $("stop").addEventListener("click", stop);
  $("next").addEventListener("click", playNextExample);
  $("live").addEventListener("click", newChat);
  $("reset").addEventListener("click", newChat);
  $("examples").addEventListener("click", openExamples);
  $("lab").addEventListener("click", openLab);
  $("go-live").addEventListener("click", closePreflight);
  $("go-examples").addEventListener("click", () => {
    $("preflight").classList.add("hidden");
    openExamples();
  });
  document.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeOverlays));

  document.addEventListener("keydown", (event) => {
    const preflightOpen = !$("preflight").classList.contains("hidden");
    if (event.key === "Escape") {
      if (overlayOpen() && !preflightOpen) closeOverlays();
      else stop();
      return;
    }
    if (preflightOpen) {
      if (event.key === "Enter" && !$("go-live").disabled) {
        event.preventDefault();
        closePreflight();
      }
      return;
    }
    if (event.key === "F1") {
      event.preventDefault();
      open("help");
      return;
    }
    if (event.key === "Enter" && state.mode === "example" && !overlayOpen() && event.target !== input) {
      event.preventDefault();
      playNextExample();
      return;
    }
    if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
    const actions = { n: newChat, e: openExamples, m: openLab };
    const action = actions[event.key.toLowerCase()];
    if (action) {
      event.preventDefault();
      action();
    }
  });
}

async function boot() {
  stage = new Stage($("drift"), $("gauge"));
  wire();
  setBusy(false);
  pollStatus();
  try {
    state.space = await api("api/space");
    stage.setSpace(state.space);
  } catch (err) {
    notice(err.message);
  }
  try {
    state.scenarios = await api("api/scenarios");
  } catch (_) {
    state.scenarios = [];
  }
  renderChecks();
}

boot();
