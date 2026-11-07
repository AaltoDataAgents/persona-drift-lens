/* Canvas views: the drift chart (each reply's position on the Assistant Axis across the
 * conversation, as plotted by Lu et al. 2026), the Agreement gauge and the layer chart.
 */

const C = {
  ink: "#45004D",
  text: "#555555",
  label: "#6E6E6E",
  land: "#D9CCDF",
  ridge: "#8C7796",
  guide: "rgba(140, 119, 150, 0.35)",
  pale: "#F3EEF5",
  valley: "#FFE58A",
  ball: "#FFCD00",
  rim: "#8A6500",
  number: "#4A3600",
  needle: "#D99216",
};

const FONT = "'Bricolage Grotesque', 'Atkinson Hyperlegible', Arial, sans-serif";
const EMOJI_FONT = "'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif";

// A badge for each label the page can show; one glance tells a room which persona it is.
export const PERSONA_EMOJI = {
  Assistant: "💬", teacher: "🧑‍🏫", tutor: "📝", counselor: "🫂",
  therapist: "🛋️", coach: "📣", mentor: "🧭", consultant: "💼",
  doctor: "🩺", lawyer: "⚖️", scientist: "🔬", philosopher: "🤔",
  journalist: "📰", marketer: "📢", chef: "🍳", detective: "🕵️",
  librarian: "📚", parent: "👪", grandparent: "👵", teenager: "🛹",
  caregiver: "🤲", romantic: "💌", optimist: "🌞", cynic: "🙄", rebel: "✊",
  gossip: "🗣️", narcissist: "🪞", celebrity: "🌟", influencer: "🤳",
  novelist: "📖", poet: "🪶", comedian: "😂", actor: "🎭", bard: "🎻",
  pirate: "🏴‍☠️", spy: "🕶️", soldier: "🪖", warrior: "⚔️",
  criminal: "🦹", hacker: "💻", mystic: "🔮", prophet: "📜", witch: "🧙",
  ghost: "👻", vampire: "🧛", angel: "😇", demon: "😈", robot: "🤖",
  alien: "👽", trickster: "🃏", jester: "🤡", editor: "✏️", analyst: "📊",
  generalist: "🧰", oracle: "🔮", salesperson: "📢", confidant: "🫂", troll: "😈", hermit: "🏔️", pilgrim: "🥾",
  whale: "🐋", gamer: "🎮", hive: "🐝", eldritch: "🐙", egregore: "🌀",
};

export function personaEmoji(name) {
  if (!name) return "";
  const first = name.split(" or ")[0];
  return PERSONA_EMOJI[first] || "🎭";
}

// The chart's colour scale, after the paper's colouring of roles by their position on the axis.
const STOPS = [
  [1.0, [255, 201, 60]], // the Assistant: sun
  [0.0, [201, 182, 228]], // the average character: lavender
  [-1.2, [42, 27, 92]], // far from the Assistant: indigo
];
function mix(a, b, t) { return a.map((v, i) => Math.round(v + (b[i] - v) * t)); }
export function axisRgb(v) {
  if (v >= STOPS[0][0]) return STOPS[0][1];
  for (let i = 0; i < STOPS.length - 1; i++) {
    const [v0, c0] = STOPS[i];
    const [v1, c1] = STOPS[i + 1];
    if (v <= v0 && v >= v1) return mix(c0, c1, (v0 - v) / (v0 - v1));
  }
  return STOPS[STOPS.length - 1][1];
}
export function axisColour(v, alpha = 1) {
  const [r, g, b] = axisRgb(v);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
// Ink or white, whichever reads on that colour.
function onColour(v) {
  const [r, g, b] = axisRgb(v);
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? C.ink : "#FFFFFF";
}
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function fit(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(width * ratio));
  const h = Math.max(1, Math.round(height * ratio));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, w: width, h: height };
}

function quantile(sorted, p) {
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))))];
}

function rampColour(t) {
  // Fade earlier replies towards pale yellow.
  const a = [0xff, 0xf6, 0xcc];
  const b = [0xff, 0xcd, 0x00];
  const m = a.map((v, i) => Math.round(v + (b[i] - v) * Math.min(1, Math.max(0, t))));
  return `rgb(${m[0]},${m[1]},${m[2]})`;
}

function ball(ctx, x, y, r, fill, label, opts = {}) {
  ctx.save();
  if (opts.live) {
    // A second ring marks the reply being written.
    ctx.beginPath();
    ctx.arc(x, y, r + 5, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(138, 101, 0, 0.45)";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.shadowColor = "rgba(255, 205, 0, 0.6)";
    ctx.shadowBlur = r * 1.2;
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.lineWidth = Math.max(1.5, r / 8);
  ctx.strokeStyle = C.rim;
  ctx.stroke();
  if (label !== undefined && label !== null) {
    ctx.fillStyle = C.number;
    ctx.font = `bold ${Math.round(r * 1.05)}px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(label), x, y + 1);
  }
  ctx.restore();
}

function haloText(ctx, text, x, y, colour) {
  ctx.lineJoin = "round";
  ctx.lineWidth = 5;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = colour;
  ctx.fillText(text, x, y);
}

export class Stage {
  constructor(drift, gauge) {
    this.canvases = { drift, gauge };
    this.space = null;
    this.turns = [];
    this.live = null; // {s, t, c}
    this.liveDrawn = null; // eased position of the live reply
    this.highlight = null;
    this.frame = null;
    this.landed = null; // {index, at}: the newest finished reply settling into place
    const observer = new ResizeObserver(() => this.draw());
    Object.values(this.canvases).forEach((c) => observer.observe(c));
    if (document.fonts) document.fonts.ready.then(() => this.draw());
  }

  setSpace(space) {
    this.space = space;
    const defaults = [...space.default_samples].sort((a, b) => a - b);
    // The usual range: where the middle 80% of the model's ordinary replies land.
    this.usual = [quantile(defaults, 0.1), quantile(defaults, 0.9)];
    this.assistant = quantile(defaults, 0.5);
    // How much a single ordinary Assistant reply varies (standard deviation): drawn around each point,
    // so differences smaller than this are not read as drift.
    const mean = defaults.reduce((a, b) => a + b, 0) / defaults.length;
    this.noise = Math.sqrt(defaults.reduce((a, b) => a + (b - mean) ** 2, 0) / defaults.length);
    // The voices of the teaching talks, under the talks' names (salesperson, confidant, troll).
    this.voices = Object.entries(space.voices).map(([name, v]) => ({ role: name, s: v.scale }));
    const scales = this.voices.map((v) => v.s);
    this.top = Math.max(quantile(defaults, 0.97), 1.3) + 0.1;
    this.bottom = Math.min(...scales, -1.0) - 0.15;
    this.draw();
  }

  setTurns(turns) {
    const grew = turns.length > this.turns.length;
    this.turns = turns;
    if (grew && !reduceMotion) {
      // One orchestrated moment: a finished reply settles onto the chart.
      this.landed = { index: turns.length - 1, at: performance.now() };
      const step = () => {
        this.draw();
        if (this.landed && performance.now() - this.landed.at < 420) requestAnimationFrame(step);
        else { this.landed = null; this.draw(); }
      };
      requestAnimationFrame(step);
    } else {
      this.draw();
    }
  }

  setLive(live) {
    this.live = live;
    if (live && this.liveDrawn === null) this.liveDrawn = live.s;
    if (!live) this.liveDrawn = null;
    this.animate();
  }

  setHighlight(role) {
    this.highlight = role;
    this.draw();
  }

  animate() {
    if (this.frame) return;
    const step = () => {
      this.frame = null;
      if (this.live && this.liveDrawn !== null) {
        const target = this.live.s;
        const gap = target - this.liveDrawn;
        this.liveDrawn = reduceMotion || Math.abs(gap) < 0.002 ? target : this.liveDrawn + gap * 0.18;
        this.draw();
        if (this.liveDrawn !== target) this.frame = requestAnimationFrame(step);
      } else {
        this.draw();
      }
    };
    this.frame = requestAnimationFrame(step);
  }

  draw() {
    if (!this.space) return;
    this.drawDrift();
    this.drawGauge();
  }

  // Drift across the conversation, as in Lu et al. (2026): one point per reply, its position on
  // the Assistant Axis (the mean over the reply's tokens), the Assistant at the top. The plot is
  // tinted by position on the axis, the way the paper colours roles.
  drawDrift() {
    const { ctx, w, h } = fit(this.canvases.drift);
    ctx.clearRect(0, 0, w, h);
    const fontPx = Math.round(Math.max(12, Math.min(18, h * 0.033)));
    ctx.font = `700 ${fontPx}px ${FONT}`;
    const labelWidth = Math.max(ctx.measureText("the Assistant").width, ctx.measureText("average character").width);
    const left = fontPx * 2.4 + labelWidth + 14;
    const right = w - Math.max(110, fontPx * 8.5);
    const top = fontPx * 2.6;
    const bottom = h - fontPx * 3.6;
    const values = [...this.turns.map((t) => t.s), ...(this.live ? [this.liveDrawn] : [])];
    const hi = Math.max(this.top, ...values.map((v) => v + 0.2));
    const lo = Math.min(this.bottom, ...values.map((v) => v - 0.2));
    const y = (v) => top + ((hi - v) / (hi - lo)) * (bottom - top);
    const slots = Math.max(8, this.turns.length + (this.live ? 1 : 0));
    const x = (i) => left + ((i + 0.5) / slots) * (right - left);

    // The colour field: sun at the Assistant, lavender at the average character, indigo below.
    const field = ctx.createLinearGradient(0, top, 0, bottom);
    for (const v of [hi, 1.0, 0.5, 0.0, -0.6, lo]) {
      const t = (y(v) - top) / (bottom - top);
      if (t >= 0 && t <= 1) field.addColorStop(t, axisColour(v, 0.32));
    }
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(left, top, right - left, bottom - top, 14);
    ctx.fillStyle = field;
    ctx.fill();
    ctx.clip();
    // The Assistant's usual range: a brighter stripe.
    ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx.fillRect(left, y(this.usual[1]), right - left, y(this.usual[0]) - y(this.usual[1]));
    ctx.restore();

    // Reference lines: the Assistant's typical reply and the average character.
    const reference = (v, text, strong) => {
      ctx.strokeStyle = strong ? C.ink : "rgba(69, 0, 77, 0.45)";
      ctx.lineWidth = strong ? 1.8 : 1.3;
      ctx.setLineDash([7, 6]);
      ctx.beginPath();
      ctx.moveTo(left, y(v));
      ctx.lineTo(right, y(v));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `${strong ? 700 : 500} ${fontPx}px ${FONT}`;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      haloText(ctx, text, left - 12, y(v), strong ? C.ink : C.label);
    };
    reference(this.assistant, "the Assistant", true);
    reference(0, "average character", false);
    ctx.font = `500 ${Math.round(fontPx * 0.85)}px ${FONT}`;
    ctx.textAlign = "right";
    haloText(ctx, "usual range", left - 12, y(this.usual[0]) - fontPx * 0.1, C.label);

    // Familiar roles on the right edge, with their badges.
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const marks = [...this.voices].sort((a, b) => b.s - a.s);
    let lastY = -Infinity;
    for (const v of marks) {
      const vy = y(v.s);
      const ly = Math.max(vy, lastY + fontPx * 1.3);
      lastY = ly;
      ctx.strokeStyle = "rgba(69, 0, 77, 0.35)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(right, vy);
      ctx.lineTo(right + 8, vy);
      ctx.lineTo(right + 14, ly);
      ctx.stroke();
      ctx.font = `500 ${Math.round(fontPx * 0.9)}px ${FONT}`;
      haloText(ctx, v.role, right + 18, ly, v.role === this.highlight ? C.ink : C.label);
    }

    // Reply numbers and the axis title.
    ctx.font = `500 ${Math.round(fontPx * 0.9)}px ${FONT}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let i = 0; i < slots; i++) haloText(ctx, String(i + 1), x(i), bottom + 8, C.label);
    haloText(ctx, "reply", (left + right) / 2, bottom + fontPx * 1.8, C.label);
    ctx.save();
    ctx.translate(fontPx * 0.9, (top + bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillStyle = C.label;
    ctx.fillText("further from the Assistant  ←  position  →  closer to the Assistant", 0, 0);
    ctx.restore();

    const points = this.turns.map((t, i) => ({ x: x(i), y: y(t.s), s: t.s, index: t.index, name: t.closest }));
    if (this.live && this.liveDrawn !== null) {
      points.push({ x: x(this.turns.length), y: y(this.liveDrawn), s: this.liveDrawn, index: this.turns.length + 1, live: true, name: this.highlight });
    }
    const r = Math.max(13, Math.min(19, ((right - left) / slots) * 0.22));

    // How much one ordinary reply varies, as a soft bar behind each point.
    const sd = Math.abs(y(0) - y(this.noise));
    ctx.strokeStyle = "rgba(69, 0, 77, 0.28)";
    ctx.lineWidth = 1.5;
    for (const p of points) {
      ctx.beginPath();
      ctx.moveTo(p.x, p.y - sd);
      ctx.lineTo(p.x, p.y + sd);
      ctx.moveTo(p.x - 4, p.y - sd);
      ctx.lineTo(p.x + 4, p.y - sd);
      ctx.moveTo(p.x - 4, p.y + sd);
      ctx.lineTo(p.x + 4, p.y + sd);
      ctx.stroke();
    }

    // The path: a smooth curve through the replies, white underneath so it reads on any colour.
    const trace = (move) => {
      points.forEach((p, i) => {
        if (i === 0) { move ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y); return; }
        const a = points[i - 1];
        const mx = (a.x + p.x) / 2;
        ctx.bezierCurveTo(mx, a.y, mx, p.y, p.x, p.y);
      });
    };
    const curve = () => { ctx.beginPath(); trace(true); };

    // The drift itself: the area between the Assistant's line and the path, in the colours of
    // the positions it covers.
    if (points.length > 1) {
      const ay = y(this.assistant);
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(points[0].x, ay);
      trace(false);
      ctx.lineTo(points[points.length - 1].x, ay);
      ctx.closePath();
      const shade = ctx.createLinearGradient(0, top, 0, bottom);
      for (const v of [hi, 1.0, 0.5, 0.0, -0.6, lo]) {
        const t = (y(v) - top) / (bottom - top);
        if (t >= 0 && t <= 1) shade.addColorStop(t, axisColour(v, 0.55));
      }
      ctx.fillStyle = shade;
      ctx.save();
      ctx.beginPath();
      ctx.rect(left, ay, right - left, bottom - ay);
      ctx.clip();
      ctx.beginPath();
      ctx.moveTo(points[0].x, ay);
      trace(false);
      ctx.lineTo(points[points.length - 1].x, ay);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
      ctx.restore();
    }
    ctx.lineCap = "round";
    curve();
    ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
    ctx.lineWidth = 4.5;
    ctx.stroke();
    curve();
    ctx.strokeStyle = "rgba(69, 0, 77, 0.75)";
    ctx.lineWidth = 2.6;
    ctx.stroke();

    // Points in the colour of their position; the newest finished one settles in.
    points.forEach((p, i) => {
      let rr = r;
      if (this.landed && this.landed.index === i) {
        const t = Math.min(1, (performance.now() - this.landed.at) / 420);
        rr = r * (1 + 0.35 * Math.sin(Math.PI * t) * (1 - t));
      }
      ctx.save();
      if (p.live) {
        ctx.shadowColor = axisColour(p.s, 0.9);
        ctx.shadowBlur = r * 1.6;
      }
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.live ? rr * 1.15 : rr, 0, Math.PI * 2);
      ctx.fillStyle = axisColour(p.s);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 2;
      ctx.strokeStyle = C.ink;
      ctx.stroke();
      if (p.live) {
        ctx.setLineDash([4, 3]);
        ctx.beginPath();
        ctx.arc(p.x, p.y, rr * 1.15 + 5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(69, 0, 77, 0.5)";
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.fillStyle = onColour(p.s);
      ctx.font = `700 ${Math.round(rr)}px ${FONT}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(p.index), p.x, p.y + 1);
      ctx.restore();
    });
  }


  drawGauge() {
    const { ctx, w, h } = fit(this.canvases.gauge);
    ctx.clearRect(0, 0, w, h);
    const fontPx = Math.round(Math.max(12, Math.min(17, h * 0.2)));
    const trackY = h * 0.5;
    const x0 = w * 0.03;
    const x1 = w * 0.97;
    const x = (v) => x0 + ((Math.max(-0.25, Math.min(1.25, v)) + 0.25) / 1.5) * (x1 - x0);
    // Honest in mint, flattering in pink.
    const grad = ctx.createLinearGradient(x0, 0, x1, 0);
    grad.addColorStop(0, "#7FD1B9");
    grad.addColorStop(0.5, "#F3EEF5");
    grad.addColorStop(1, "#E0457B");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x0, trackY - 6, x1 - x0, 12, 6);
    ctx.fill();
    ctx.font = `500 ${fontPx}px ${FONT}`;
    ctx.fillStyle = C.label;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText("honest", x0, trackY + 16);
    ctx.textAlign = "right";
    ctx.fillText("flattering", x1, trackY + 16);

    // Earlier replies as small ticks, the latest finished reply as a numbered dot; the reply
    // being written as the needle.
    const r = Math.max(7, Math.min(10, h * 0.13));
    const n = this.turns.length;
    ctx.strokeStyle = "rgba(138, 101, 0, 0.45)";
    ctx.lineWidth = 2;
    this.turns.slice(0, this.live ? n : n - 1).forEach((t) => {
      ctx.beginPath();
      ctx.moveTo(x(t.t), trackY - 8);
      ctx.lineTo(x(t.t), trackY + 8);
      ctx.stroke();
    });
    if (n && !this.live) {
      const t = this.turns[n - 1];
      ball(ctx, x(t.t), trackY, r, C.ball, t.index);
    }
    if (this.live) {
      const px = x(this.live.t);
      ctx.fillStyle = C.needle;
      ctx.beginPath();
      ctx.moveTo(px, trackY - r - 2);
      ctx.lineTo(px - 8, trackY - r - 15);
      ctx.lineTo(px + 8, trackY - r - 15);
      ctx.closePath();
      ctx.fill();
    }
  }


  drawProfiles(canvas, turns, readingLayer) {
    const { ctx, w, h } = fit(canvas);
    ctx.clearRect(0, 0, w, h);
    const withProfile = turns.filter((t) => t.profile);
    const layers = withProfile.length ? withProfile[0].profile.length : this.space.n_layers;
    const padL = 70;
    const padB = 30;
    const cw = (w - padL - 10) / Math.max(8, withProfile.length);
    const ch = (h - padB - 6) / layers;
    ctx.font = `13px ${FONT}`;
    ctx.fillStyle = C.label;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let l = 0; l < layers; l += 5) ctx.fillText(`layer ${l}`, padL - 8, 6 + (layers - 1 - l + 0.5) * ch);
    withProfile.forEach((turn, i) => {
      turn.profile.forEach((v, l) => {
        // 1 (Assistant) is pale yellow, 0 and below deepen to aubergine.
        const t = Math.max(0, Math.min(1, (1.2 - v) / 1.8));
        const a = [0xff, 0xe5, 0x8a];
        const b = [0x45, 0x00, 0x4d];
        const m = a.map((c, k) => Math.round(c + (b[k] - c) * t));
        ctx.fillStyle = `rgb(${m[0]},${m[1]},${m[2]})`;
        ctx.fillRect(padL + i * cw + 1, 6 + (layers - 1 - l) * ch, cw - 2, ch);
      });
      ctx.fillStyle = C.label;
      ctx.textAlign = "center";
      ctx.fillText(String(turn.index), padL + (i + 0.5) * cw, h - padB / 2);
    });
    const mark = (l, colour, dash) => {
      ctx.strokeStyle = colour;
      ctx.setLineDash(dash);
      ctx.lineWidth = 2;
      ctx.strokeRect(padL - 2, 6 + (layers - 1 - l) * ch, w - padL - 8, ch);
      ctx.setLineDash([]);
    };
    mark(readingLayer, C.ink, []);
  }
}
