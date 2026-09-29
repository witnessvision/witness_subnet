"use strict";
// Read-only subnet dashboard. Every value comes from the backend contract
// (witness-dashboard-2); nothing here computes consensus, queue order or king.
// Untrusted strings are only ever written with textContent or attributes.
(() => {
  const $ = (selector) => document.querySelector(selector);
  const el = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text != null) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  const known = (v) => v !== null && v !== undefined;
  const finite = (n) => typeof n === "number" && Number.isFinite(n);
  const integer = (n) => known(n) && Number.isFinite(n) ? new Intl.NumberFormat("en-US").format(n) : "Unknown";
  const decimal = (n, digits = 3) => known(n) && Number.isFinite(n) ? n.toFixed(digits) : "—";
  const stake = (n) => known(n) && Number.isFinite(n) ? new Intl.NumberFormat("en-US", {maximumFractionDigits: 2}).format(n) : "Unknown";
  const short = (s) => s.length > 18 ? `${s.slice(0, 8)}…${s.slice(-6)}` : s;
  const list = (v) => Array.isArray(v) ? v : [];
  const pct = (f) => finite(f) ? `${Math.round(f * 1000) / 10}%` : "?";
  const clock = (ms) => new Date(ms).toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  const age = (s) => s < 60 ? `${s} s` : s < 3600 ? `${Math.floor(s / 60)} min` : `${Math.floor(s / 3600)} h`;
  const plural = (n, word) => `${integer(n)} ${word}${n === 1 ? "" : "s"}`;
  const clamp = (x) => Math.max(0, Math.min(1, x));
  const unknown = (text = "Unknown") => el("span", text, "sn-unknown");

  async function read(path) {
    const response = await fetch(path, {cache: "no-store", headers: {Accept: "application/json"}, signal: AbortSignal.timeout(15000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  async function copy(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch { /* fall back below */ }
    const area = el("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.className = "sn-offscreen";
    document.body.append(area);
    area.select();
    let done = false;
    try { done = document.execCommand("copy"); } catch { done = false; }
    area.remove();
    return done;
  }

  // Identifier with a middle-truncated (or full) visible form and a button that copies the full value.
  function ident(value, label, full = false) {
    if (!known(value) || value === "") return el("span", "Unknown", "sn-unknown");
    const text = String(value);
    const wrap = el("span", null, full ? "sn-id sn-id-full" : "sn-id");
    const code = el("code", full ? text : short(text));
    code.title = text;
    const button = el("button", "Copy", "sn-copy");
    button.type = "button";
    button.setAttribute("aria-label", `Copy full ${label}`);
    button.onclick = async () => {
      button.textContent = await copy(text) ? "Copied" : "Failed";
      setTimeout(() => { button.textContent = "Copy"; }, 1400);
    };
    wrap.append(code, button);
    return wrap;
  }

  // A UID is shown only when the backend supplies it for this exact row; it is never looked up from a hotkey.
  function who(uid, value, label) {
    if (!finite(uid)) return ident(value, label);
    const box = el("span", null, "sn-who");
    box.append(el("strong", `UID ${uid}`, "sn-uid"), ident(value, label));
    return box;
  }

  const TONES = {
    good: ["fresh", "applied", "consumed", "evaluated", "done", "complete", "completed", "closed", "success", "finalized", "included", "won"],
    warn: ["stale", "reserved", "queued", "pending", "waiting", "retry", "open", "evaluating"],
    live: ["running"],
    bad: ["unreachable", "failed", "error", "rejected", "invalid", "expired", "lost", "timeout"],
  };
  function badge(value) {
    if (!known(value) || value === "") return el("span", "Unknown", "sn-unknown");
    const text = String(value);
    const key = text.toLowerCase();
    const tone = Object.keys(TONES).find((t) => TONES[t].includes(key)) || "neutral";
    return el("span", text, `sn-badge sn-${tone}`);
  }

  // Relative age that keeps ticking; `unix` is seconds. Older than the backend's 120 s freshness window reads as old.
  function since(unix, prefix = "") {
    if (!finite(unix)) return unknown(`${prefix}time unknown`);
    const span = el("span", null, "sn-since");
    span.dataset.since = String(unix);
    span.dataset.prefix = prefix;
    tick(span);
    return span;
  }
  function tick(span) {
    const s = Math.max(0, Math.round(Date.now() / 1000 - Number(span.dataset.since)));
    span.textContent = `${span.dataset.prefix}${age(s)} ago`;
    span.classList.toggle("sn-old", s > 120);
  }

  // Evaluator telemetry stages. Anything else is shown verbatim, never guessed.
  const STAGES = {
    waiting: ["Waiting", "sn-neutral"], preparing: ["Preparing batch", "sn-live"], downloading: ["Downloading weights", "sn-live"],
    evaluating: ["Running clips", "sn-live"], judging: ["Scoring claims", "sn-live"], deferred: ["Deferred", "sn-warn"],
  };
  const WORKING = ["preparing", "downloading", "evaluating", "judging"];
  function stageBadge(stage) {
    if (!known(stage) || stage === "") return unknown();
    const [label, tone] = STAGES[stage] || [String(stage), "sn-neutral"];
    const node = el("span", label, `sn-badge ${tone}`);
    if (stage === "deferred") node.title = "Infrastructure deferral: retried later, never scored as zero";
    return node;
  }

  function clipProgress(done, total) {
    if (!finite(done) || !finite(total) || total <= 0) return unknown("Clip progress unknown");
    const box = el("div", null, "sn-progress");
    const bar = el("progress", null, "sn-bar");
    bar.max = total;
    bar.value = Math.min(done, total);
    bar.setAttribute("aria-label", `${done} of ${total} clips completed`);
    box.append(bar, el("span", `${integer(done)} of ${integer(total)} clips`));
    return box;
  }

  // Weight vectors are shown as published: UIDs with normalized shares.
  function vector(uids, weights) {
    if (!Array.isArray(uids)) return unknown("Vector not reported");
    if (!uids.length) return unknown("Empty vector");
    const box = el("span", null, "sn-vector");
    uids.forEach((uid, i) => box.append(el("span", `UID ${uid} · ${pct(Array.isArray(weights) ? weights[i] : null)}`, "sn-chip")));
    return box;
  }
  function sameVector(a, b) {
    if (!Array.isArray(a?.uids) || !Array.isArray(b?.uids) || a.uids.length !== b.uids.length) return false;
    return a.uids.every((uid, i) => {
      const j = b.uids.indexOf(uid);
      return j >= 0 && finite(a.weights?.[i]) && finite(b.weights?.[j]) && Math.abs(a.weights[i] - b.weights[j]) <= 0.005;
    });
  }

  // An early stop publishes upper bounds, never a measured score or a win.
  function evalStatus(value) {
    return value === "early_stop" ? el("span", "Early stop", "sn-badge sn-warn") : badge(value);
  }

  // Score with a 0–1 bar. An upper bound keeps its "≤" and a hollow bar; a missing score stays unknown, never zero.
  function meter(value, upper, digits = 4) {
    const measured = finite(value);
    const bound = !measured && finite(upper);
    const box = el("span", null, bound ? "sn-meter sn-meter-upper" : "sn-meter");
    if (!measured && !bound) {
      box.append(unknown("—"));
      box.title = "Not reported";
      return box;
    }
    const shown = measured ? value : upper;
    const bar = el("span", null, "sn-meter-bar");
    bar.setAttribute("aria-hidden", "true");
    const fill = el("i");
    fill.style.width = `${clamp(shown) * 100}%`;
    bar.append(fill);
    box.append(el("span", `${bound ? "≤ " : ""}${decimal(shown, digits)}`, "sn-meter-num"), bar);
    if (bound) {
      box.title = "Upper bound from an early stop, not a measured score";
      box.append(el("small", "upper bound", "sn-muted"));
    }
    return box;
  }

  // Queue reasons as the evaluator records them: exception class names or policy codes.
  // The original code always stays visible next to the translation.
  const REASONS = {
    ConnectionRefusedError: ["Miner unreachable", "bad", "The evaluator could not connect to the miner serving this model. Retried later; nothing was scored."],
    ValueError: ["Preparation failed", "bad", "The evaluator could not prepare this model for the batch. Retried later. The model is not marked invalid and the hotkey is not consumed."],
    TimeoutError: ["Timed out", "bad", "An evaluation step exceeded its time limit. Retried later."],
    InterruptedError: ["Interrupted", "warn", "The attempt stopped before finishing. Retried later."],
    DownloadBudgetExceeded: ["Download budget exhausted", "warn", "The model download exceeded its time budget. Parked without a result or hotkey use."],
    compression_required: ["Zstandard required", "warn", "The model must use Zstandard transport before it can be evaluated. Held without consuming the hotkey."],
    compression_ready: ["Zstandard ready", "neutral", "Zstandard transport detected; back in the queue."],
    eligible_in_next_window: ["Awaiting next window", "neutral", "Not a candidate in the current window. Eligible when the next window opens."],
    retry_after_restart: ["Retry after restart", "neutral", "Interrupted by an evaluator restart; retried without waiting."],
    window_closed_before_publication: ["Window closed first", "warn", "The window closed before the result was published. Retried later."],
    p2p_resubmission_required: ["Resubmission required", "warn", "Legacy submission. The miner must resubmit to be evaluated."],
  };
  function reasonInfo(reason) {
    if (!known(reason) || reason === "") return null;
    const raw = String(reason);
    if (Object.hasOwn(REASONS, raw)) {
      const [label, tone, text] = REASONS[raw];
      return {raw, label, tone, text};
    }
    if (/(Error|Exception)$/.test(raw)) return {raw, label: raw, tone: "bad", text: "The last attempt raised this error. The evaluator retries later."};
    return {raw, label: raw, tone: "neutral", text: null};
  }
  function reasonCell(reason) {
    const info = reasonInfo(reason);
    if (!info) return el("span", "—", "sn-muted");
    const box = el("span", null, "sn-reason");
    const top = el("span", null, "sn-reason-top");
    top.append(el("strong", info.label, `sn-${info.tone}`));
    if (info.label !== info.raw) top.append(el("code", info.raw, "sn-raw"));
    box.append(top);
    if (info.text) box.append(el("small", info.text));
    return box;
  }

  // Counts of backend-reported fields only; order and eligibility stay the backend's.
  function queueSummary(queue) {
    const reasons = new Map();
    let inLine = 0, running = 0, blocked = 0;
    for (const row of queue) {
      if (finite(row.position)) inLine++;
      if (row.status === "running") running++;
      const info = reasonInfo(row.reason);
      if (!info) continue;
      if (info.tone === "bad") blocked++;
      const entry = reasons.get(info.label) || {tone: info.tone, n: 0, raw: info.raw};
      entry.n++;
      reasons.set(info.label, entry);
    }
    const chips = el("ul", null, "sn-qsum");
    chips.setAttribute("aria-label", "Queue summary");
    const add = (n, text, tone, title) => {
      const li = el("li", null, `sn-qchip sn-${tone}`);
      li.append(el("strong", integer(n)), el("span", ` ${text}`));
      if (title) li.title = title;
      chips.append(li);
    };
    add(queue.length, "reserved", "neutral");
    if (inLine) add(inLine, "in line this window", "neutral");
    if (running) add(running, "running", "live");
    for (const [label, entry] of reasons) add(entry.n, label, entry.tone, entry.raw);
    return {chips, inLine, running, blocked, nextWindow: reasons.get(REASONS.eligible_in_next_window[0])?.n || 0};
  }

  function node(value) { return value instanceof Node ? value : el("span", value); }

  // Table that collapses into labelled cards on narrow screens.
  function table(caption, columns, rows, rowClass) {
    const wrap = el("div", null, "sn-table-wrap");
    const t = el("table", null, "sn-table");
    t.append(el("caption", caption, "sn-visually-hidden"));
    const head = el("tr");
    for (const [label, cls] of columns) {
      const th = el("th", label, cls);
      th.scope = "col";
      head.append(th);
    }
    const thead = el("thead");
    thead.append(head);
    const tbody = el("tbody");
    rows.forEach((cells, index) => {
      const tr = el("tr");
      if (rowClass) tr.className = rowClass(index);
      cells.forEach((value, i) => {
        const td = el("td", null, columns[i][1]);
        td.dataset.label = columns[i][0];
        td.append(node(value));
        tr.append(td);
      });
      tbody.append(tr);
    });
    t.append(thead, tbody);
    wrap.append(t);
    return wrap;
  }

  function empty(title, text) {
    const box = el("div", null, "sn-empty");
    box.append(el("strong", title), el("p", text));
    return box;
  }

  function failure(target, label, retry) {
    const box = el("div", null, "sn-error");
    box.append(el("strong", `${label} unavailable`), el("p", "Nothing is shown in its place. Try reading the source again."));
    const button = el("button", "Try again", "sn-btn");
    button.type = "button";
    button.onclick = retry;
    box.append(button);
    target.replaceChildren(box);
  }

  function facts(pairs) {
    const dl = el("dl", null, "sn-facts");
    for (const [label, value] of pairs) {
      const row = el("div");
      const dd = el("dd");
      dd.append(node(value));
      row.append(el("dt", label), dd);
      dl.append(row);
    }
    return dl;
  }

  const state = {data: null, validator: null, hotkeys: {state: "consumed", q: "", page: 1, seq: 0}, loading: false, fetchedAt: null, failed: false};
  const openWindowId = (data) => data?.window && data.window.state === "open" ? data.window.id : null;

  // ---------- Overview ----------
  function renderChain(data) {
    const meta = $("#chain-meta");
    meta.replaceChildren();
    const add = (label, value) => {
      const row = el("div");
      const dd = el("dd");
      dd.append(node(value));
      row.append(el("dt", label), dd);
      meta.append(row);
    };
    add("Block", known(data.block) ? integer(data.block) : unknown());
    const w = data.window;
    add("Window", w ? `${w.id ?? "?"} · ${w.state ?? "unknown"}` : unknown());
    const validators = list(data.validators);
    const fresh = validators.filter((v) => v.freshness === "fresh").length;
    add("Chain view", validators.length ? el("span", `${fresh} of ${validators.length} fresh`, fresh === validators.length ? "sn-ok" : "sn-warn") : unknown("None reporting"));
    add("Policy", known(data.policy_hash) ? ident(data.policy_hash, "policy hash") : unknown());
    $("#demo-banner").hidden = data.is_demo !== true;
    document.body.classList.toggle("sn-is-demo", data.is_demo === true);
  }

  function renderUpdated() {
    const target = $("#updated");
    if (!state.fetchedAt) { target.textContent = state.failed ? "Read failed" : "Reading…"; return; }
    const s = Math.max(0, Math.round((Date.now() - state.fetchedAt) / 1000));
    target.textContent = `${state.failed ? "Refresh failed · last read" : "Read"} ${age(s)} ago`;
    target.classList.toggle("sn-warn", state.failed);
  }

  // Stale means the page may not show the present chain state; say why instead of hiding the data.
  function renderStale() {
    const reasons = [];
    const data = state.data;
    if (data && state.failed) reasons.push(`The latest refresh failed; showing the read from ${clock(state.fetchedAt)}.`);
    if (data && !known(data.block)) reasons.push("The authoritative chain source is unavailable, so block, king and weights are unknown.");
    const validators = list(data?.validators);
    if (validators.length && !validators.some((v) => v.freshness === "fresh")) reasons.push("No reporting validator shows a caught-up chain view from the last 2 minutes.");
    $("#stale-banner").hidden = !reasons.length;
    $("#stale-text").textContent = reasons.join(" ");
  }

  // Source read errors belong to this dashboard's plumbing, not to any evaluation or queue retry.
  function renderErrors(errors) {
    const target = $("#errors");
    target.replaceChildren();
    if (!Array.isArray(errors) || !errors.length) return;
    const box = el("div", null, "sn-warnings");
    box.append(el("strong", `Dashboard could not read ${plural(errors.length, "status source")}`),
      el("p", "Read errors between this dashboard and its configured sources. They are not evaluation results or queue retry reasons.", "sn-muted sn-small"));
    const items = el("ul");
    for (const item of errors) {
      const li = el("li");
      li.append(el("code", item.source ?? "unknown source"), el("span", ` ${item.error ?? "unknown error"}`));
      items.append(li);
    }
    box.append(items);
    target.append(box);
  }

  function card(eyebrow, title, body, extra, cls = "") {
    const article = el("article", null, `sn-card ${cls}`.trim());
    const head = el("div", null, "sn-card-head");
    head.append(el("p", eyebrow, "sn-eyebrow"));
    if (extra) head.append(extra);
    article.append(head);
    if (title) article.append(title);
    for (const part of body) article.append(part);
    return article;
  }

  function splitBar(intended) {
    const burn = intended?.burn_fraction;
    const share = intended?.king_fraction;
    if (!finite(burn) || !finite(share)) return el("p", "Intended burn and king split not reported.", "sn-unknown");
    const box = el("div", null, "sn-split-wrap");
    const bar = el("div", null, "sn-split");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", `Intended weights: ${pct(burn)} burned, ${pct(share)} to the king`);
    for (const [fraction, cls] of [[burn, "sn-split-burn"], [share, "sn-split-king"]]) {
      if (fraction <= 0) continue;
      const part = el("span", null, cls);
      part.style.flexGrow = String(fraction);
      bar.append(part);
    }
    const legend = el("div", null, "sn-split-legend");
    for (const [label, fraction, uid, cls, none] of [["burn", burn, intended.burn_uid, "sn-split-burn", "burn UID unknown"], ["king", share, intended.king_uid, "sn-split-king", "no king"]]) {
      const item = el("span");
      item.append(el("i", null, `sn-key ${cls}`), el("strong", pct(fraction)), el("span", ` ${label} · ${known(uid) ? `UID ${uid}` : none}`));
      legend.append(item);
    }
    box.append(bar, legend);
    return box;
  }

  // The king's own latest published result: exact model ID only, measured before upper bound, newest window first.
  function kingResult(data) {
    const model = data.king?.model_id;
    if (!known(model)) return null;
    const newest = (rows) => rows.reduce((best, e) => !best || e.window_id > best.window_id ? e : best, null);
    const rows = list(data.evaluations).filter((e) => e.model_id === model && finite(e.window_id));
    const measured = newest(rows.filter((e) => finite(e.quality)));
    const latest = newest(rows);
    return latest && {row: measured || latest, measured: Boolean(measured), newer: measured && latest.window_id > measured.window_id ? latest : null};
  }

  function reportButton(evaluation, text, cls = "sn-link") {
    const button = el("button", text, cls);
    button.type = "button";
    button.dataset.action = "detail";
    button.setAttribute("aria-haspopup", "dialog");
    button.onclick = () => openDetail(evaluation, button);
    return button;
  }

  function kingResultBlock(data) {
    const box = el("div", null, "sn-king-result");
    const found = kingResult(data);
    if (!found) {
      box.append(el("p", "LATEST RESULT", "sn-eyebrow"), el("p", "No published result for this exact model yet.", "sn-muted sn-small"));
      return box;
    }
    const e = found.row;
    const open = openWindowId(data);
    const head = el("div", null, "sn-kr-head");
    head.append(el("p", `${found.measured ? "LATEST MEASURED RESULT" : "LATEST RESULT"} · WINDOW ${e.window_id}`, "sn-eyebrow"));
    if (e.status && e.status !== "done") head.append(evalStatus(e.status));
    head.append(e.available ? el("span", "Closed", "sn-badge sn-good") : e.window_id === open ? el("span", "Window open", "sn-badge sn-warn") : el("span", "Report unavailable", "sn-badge sn-neutral"));
    const nums = el("dl", null, "sn-kr-nums");
    for (const [label, value, upper] of [["Quality", e.quality, e.quality_upper], ["Reward", e.reward, e.reward_upper]]) {
      const row = el("div");
      const dd = el("dd");
      dd.append(meter(value, upper));
      row.append(el("dt", label), dd);
      nums.append(row);
    }
    const foot = el("div", null, "sn-kr-foot");
    const by = el("span", null, "sn-kr-by");
    by.append(el("span", "Evaluator "), who(list(data.validators).find((v) => v.hotkey === e.validator)?.uid, e.validator, "evaluator hotkey"));
    foot.append(by, e.available ? reportButton(e, `Open window ${e.window_id} report`, "sn-btn sn-btn-accent") : el("span", e.window_id === open ? "Clips open after the window closes." : "Report not available.", "sn-muted sn-small"));
    box.append(head, nums, foot);
    if (found.newer) box.append(el("p", `Window ${found.newer.window_id} has a newer result without a measured score (${found.newer.status ?? "status unknown"}); see Results.`, "sn-note"));
    return box;
  }

  function renderCrown(data) {
    const king = data.king;
    const intended = data.weights?.intended;
    const title = el("div", null, "sn-card-title");
    const body = [];
    if (!known(data.block) && !king) {
      // Without the authoritative chain projection, "no king" would be a guess.
      title.append(unknown("King unknown"));
      body.push(el("p", "The authoritative chain source is unavailable. King, burn and weights stay unknown until it answers.", "sn-lede"));
      return card("KING", title, body, el("span", "Unknown", "sn-badge sn-neutral"), "sn-crown");
    }
    if (king) {
      const uid = finite(king.uid) ? king.uid : intended?.king_uid;
      const pairs = [["Coldkey", ident(king.coldkey, "king coldkey")], ["Model", ident(king.model_id, "king model id")]];
      if (finite(uid)) {
        title.append(el("span", `UID ${uid}`, "sn-king-uid"));
        pairs.unshift(["Hotkey", ident(king.hotkey, "king hotkey")]);
      } else {
        title.append(ident(king.hotkey, "king hotkey"));
      }
      body.push(facts(pairs), kingResultBlock(data));
    } else {
      title.append(el("span", "No king yet"));
      const burn = finite(intended?.burn_fraction) ? `${pct(intended.burn_fraction)} of the weight burns` : "Weight burns";
      const uid = known(intended?.burn_uid) ? `UID ${intended.burn_uid}` : "the burn UID";
      body.push(el("p", `${burn} to ${uid} until the chain crowns a first king. The first king comes from a bootstrap pair, so the subnet waits for two valid submissions evaluated on the same batch.`, "sn-lede"));
    }
    body.push(el("p", "WEIGHT SPLIT · INTENDED", "sn-eyebrow sn-sub-eyebrow"), splitBar(intended));
    body.push(el("p", "Read from finalized chain consensus, never inferred from scores.", "sn-note"));
    const status = king ? el("span", "Crowned", "sn-badge sn-good") : el("span", "No king · burning", "sn-badge sn-warn");
    return card("KING", title, body, status, "sn-crown");
  }

  // Submission receipts: "submitted" proves only that the extrinsic was sent, not that weights applied.
  const SUBMITTED = {
    submitted: ["Submitted · unconfirmed", "sn-warn"], submitting: ["Submitting", "sn-live"], applied: ["Applied", "sn-good"],
    rejected: ["Rejected", "sn-bad"], unknown: ["Outcome unknown", "sn-warn"],
  };
  function submittedBadge(status) {
    if (!known(status) || status === "") return unknown();
    const [label, tone] = SUBMITTED[status] || [String(status), "sn-neutral"];
    const node = el("span", label, `sn-badge ${tone}`);
    node.title = `Recorded status: ${status}`;
    return node;
  }

  function renderWeights(weights) {
    const {decision, intended, submitted, applied} = weights;
    const steps = el("ol", null, "sn-steps");
    const step = (label, item, hint, fill, extra, missing = "None reported") => {
      const li = el("li", null, item ? "" : "sn-step-missing");
      const top = el("div", null, "sn-step-top");
      top.append(el("strong", label));
      if (extra) top.append(...extra.filter(Boolean));
      li.append(top);
      if (item) fill(li); else li.append(unknown(missing));
      li.append(el("small", hint, "sn-step-hint"));
      steps.append(li);
    };
    step("Decision", decision, "Winner computed from finalized commitments", (li) => {
      li.append(known(decision.hotkey) ? ident(decision.hotkey, "decision hotkey") : el("span", "No king · burn", "sn-strong"), el("small", `Block ${integer(decision.block)}`));
      const open = list(decision.inconclusive).length;
      if (open) li.append(el("small", `${open} inconclusive: early-stop evidence was incomplete and could not decide a promotion; unresolved evaluators may retry with the next window's fresh batch`, "sn-warn"));
    });
    step("Intended", intended, "Vector this validator intends to set", (li) => li.append(vector(intended.uids, intended.weights)),
      [intended ? badge(intended.source) : null]);
    step("Submitted", submitted, "A submission does not prove the weights applied; commit/reveal can delay application", (li) => {
      if (Array.isArray(submitted.uids)) li.append(vector(submitted.uids, submitted.weights));
      li.append(el("small", `Block ${integer(submitted.block)}`));
      if (known(submitted.receipt)) li.append(responseBlock("Receipt", submitted.receipt));
    }, [submitted ? submittedBadge(submitted.status) : null,
      submitted && intended && Array.isArray(submitted.uids) && !sameVector(submitted, intended) ? el("span", "differs from intended", "sn-badge sn-warn") : null]);
    step("Applied", applied, "Observed in finalized chain state. Until then it is unconfirmed, not zero", (li) => {
      if (known(applied.hotkey)) li.append(ident(applied.hotkey, "applied king hotkey"));
      li.append(vector(applied.uids, applied.weights), el("small", `Block ${integer(applied.block)}`));
    }, [applied && intended ? (sameVector(applied, intended) ? el("span", "matches intended", "sn-badge sn-good") : el("span", "differs from intended", "sn-badge sn-warn")) : null],
    "Unconfirmed · not observed on finalized chain");
    return card("WEIGHTS · DECISION → APPLIED", null, [steps], null, "sn-card-wide sn-weights");
  }

  function modelRole(modelId, king) {
    if (!known(modelId)) return "No model";
    if (king && king.model_id === modelId) return "King baseline";
    return king ? "Challenger" : "Bootstrap candidate";
  }

  // One sentence that separates work from chain freshness and queue trouble; a waiting evaluator with blocked rows is not "fine".
  function headline(p, queue, summary) {
    if (!p) return ["No telemetry", "neutral", "This validator publishes no progress, so what it is doing is unknown."];
    if (WORKING.includes(p.stage)) return [`Working · ${STAGES[p.stage][0]}`, "live", `${modelRole(p.model_id, state.data?.king)} in window ${p.window_id ?? "?"}.`];
    if (p.stage === "deferred") return ["Last attempt deferred", "warn", "Telemetry reports an infrastructure deferral; the challenge is retried later, never scored as zero."];
    if (p.stage !== "waiting") return [String(p.stage ?? "Unknown stage"), "neutral", "Stage reported by the evaluator as is."];
    if (!summary) return ["Waiting", "neutral", "Telemetry reports no evaluation in progress. The queue is not reported."];
    if (summary.blocked) return ["Waiting · queue held by errors", "bad", `${summary.blocked} of ${plural(queue.length, "reserved challenge")} held by retry errors. Telemetry reports no evaluation in progress.`];
    if (!queue.length) return ["Waiting · queue empty", "neutral", "No reserved challenge for this evaluator."];
    if (summary.nextWindow === queue.length) return ["Waiting for next window", "neutral", "Every reserved challenge becomes eligible when the next window opens."];
    return ["Waiting", "neutral", "Telemetry reports no evaluation in progress."];
  }

  function nowTile(v, data) {
    const p = v.progress;
    const queue = data.queues ? data.queues[v.hotkey] : undefined;
    const summary = Array.isArray(queue) ? queueSummary(queue) : null;
    const tile = el("article", null, "sn-now");
    const head = el("div", null, "sn-now-head");
    head.append(who(v.uid, v.hotkey, "validator hotkey"), el("span", v.mode ?? "unknown mode", "sn-muted sn-small"));
    let [title, tone, detail] = headline(p, queue, summary);
    // Old telemetry or a stale chain view says what was last reported, not what happens now.
    if (p && (v.freshness !== "fresh" || !finite(p.updated_unix) || Date.now() / 1000 - p.updated_unix > 120)) {
      [title, tone, detail] = ["Current activity unknown", "warn", `Telemetry or chain view is stale. Last report: ${title}.`];
    }
    tile.append(head, el("p", title, `sn-now-title sn-${tone}`), el("p", detail, "sn-now-detail"));
    const signals = el("dl", null, "sn-signals");
    const signal = (label, ...parts) => {
      const row = el("div");
      const dd = el("dd");
      dd.append(...parts);
      row.append(el("dt", label), dd);
      signals.append(row);
    };
    signal("Chain", badge(v.freshness), el("span", `block ${integer(v.block)}`, "sn-muted sn-small sn-mono"));
    if (p) {
      const work = [stageBadge(p.stage), el("span", known(p.window_id) ? `window ${p.window_id}` : "window unknown", "sn-muted sn-small sn-mono")];
      if (known(p.model_id)) {
        const model = el("span", null, "sn-inline");
        model.append(el("span", modelRole(p.model_id, data.king), "sn-role"), ident(p.model_id, "model id"));
        work.push(model);
      }
      if (WORKING.includes(p.stage)) work.push(clipProgress(p.completed_clips, p.total_clips));
      work.push(since(p.updated_unix, "telemetry "));
      signal("Work", ...work);
    } else {
      signal("Work", unknown("No progress reported"));
    }
    signal("Queue", summary ? summary.chips : unknown("Queue not reported"));
    tile.append(signals);
    return tile;
  }

  function renderNow(data) {
    const validators = list(data.validators).filter((v) => v.mode === "evaluator" || known(v.progress));
    const box = el("div", null, "sn-now-list");
    if (!validators.length) box.append(empty("No evaluator telemetry", "No validator publishes progress to this dashboard. Telemetry is opt-in, so this says nothing about evaluations or commitments on chain."));
    for (const v of validators) box.append(nowTile(v, data));
    const note = el("p", "Signed telemetry, display only. It never decides the king or the weights.", "sn-note");
    return card("EVALUATION NOW", null, [box, note], null, "sn-now-card");
  }

  function renderWindow(data) {
    const w = data.window;
    const title = el("div", null, "sn-card-title");
    title.append(el("span", w ? `Window ${w.id ?? "?"}` : "No window reported"));
    const body = w ? [facts([
      ["Epochs", known(w.start_epoch) ? `${w.start_epoch} – ${known(w.end_epoch) ? w.end_epoch : "?"}` : "Unknown"],
      ["Start block", integer(w.start_block)],
      ["End block", known(w.end_block) ? integer(w.end_block) : el("span", "Pending finalization", "sn-unknown")],
    ])] : [el("p", "No evaluation window is reported.", "sn-note")];
    if (w && w.state === "open") body.push(el("p", "Clip detail for this window stays hidden until it closes. Closed windows open from Results.", "sn-note"));
    const jump = el("a", "Go to results ↓", "sn-link");
    jump.href = "#evaluations";
    body.push(jump);
    return card("WINDOW · 2 EPOCHS", title, body, w ? badge(w.state) : null, "sn-window");
  }

  function renderOverview(data) {
    const target = $("#overview-body");
    target.replaceChildren(renderCrown(data), renderNow(data), renderWindow(data), renderWeights(data.weights || {}));
    target.setAttribute("aria-busy", "false");
  }

  // ---------- Validators ----------
  function progressCell(p) {
    if (!p) return unknown();
    const box = el("span", null, "sn-stack");
    const top = el("span", null, "sn-inline");
    top.append(stageBadge(p.stage));
    if (finite(p.completed_clips) && finite(p.total_clips)) top.append(el("small", `${p.completed_clips}/${p.total_clips} clips`, "sn-muted"));
    box.append(top, since(p.updated_unix));
    return box;
  }

  function chainCell(v) {
    const box = el("span", null, "sn-stack");
    box.append(badge(v.freshness), el("small", `block ${integer(v.block)}`, "sn-muted sn-mono"));
    return box;
  }

  function renderValidators(data) {
    const target = $("#validators-body");
    const validators = list(data.validators);
    if (!validators.length) {
      target.replaceChildren(empty("No validator reporting", "No validator publishes its status to this dashboard. Reporting is opt-in: registered validators may still hold chain commitments that are not listed here."));
      return;
    }
    const rows = validators.map((v) => {
      const select = el("button", v.hotkey === state.validator ? "Selected" : "Inspect", "sn-link");
      select.type = "button";
      select.setAttribute("aria-pressed", String(v.hotkey === state.validator));
      select.onclick = () => { selectValidator(v.hotkey); $("#queue").scrollIntoView({block: "start"}); };
      return [who(v.uid, v.hotkey, "validator hotkey"), badge(v.mode), stake(v.stake), chainCell(v), known(v.commitment_block) ? integer(v.commitment_block) : el("span", "None", "sn-unknown"), progressCell(v.progress), integer(v.used_count), integer(v.reserved_count), select];
    });
    target.replaceChildren(table("Validators", [["Validator"], ["Mode"], ["Stake", "sn-num"], ["Chain view"], ["Commitment block", "sn-num"], ["Work"], ["Consumed", "sn-num"], ["Reserved", "sn-num"], ["", "sn-action"]], rows,
      (i) => validators[i].hotkey === state.validator ? "sn-selected" : ""));
  }

  function renderScope(data) {
    const select = $("#validator-select");
    const validators = list(data.validators);
    select.replaceChildren();
    for (const v of validators) {
      const option = el("option", `${finite(v.uid) ? `UID ${v.uid} · ` : ""}${short(String(v.hotkey))} · ${v.mode ?? "unknown"}`);
      option.value = v.hotkey;
      select.append(option);
    }
    if (!validators.length) select.append(el("option", "None reporting"));
    select.disabled = !validators.length;
    if (state.validator) select.value = state.validator;
    const scope = $("#scope-id");
    scope.replaceChildren();
    const v = currentValidator();
    if (v) scope.append(ident(v.hotkey, "selected validator hotkey"), badge(v.freshness));
  }

  const currentValidator = () => (state.data?.validators || []).find((v) => v.hotkey === state.validator) || null;

  function defaultValidator(validators) {
    const fromUrl = new URLSearchParams(location.search).get("validator");
    if (fromUrl && validators.some((v) => v.hotkey === fromUrl)) return fromUrl;
    if (state.validator && validators.some((v) => v.hotkey === state.validator)) return state.validator;
    const pick = validators.find((v) => v.mode === "evaluator" && v.freshness === "fresh") || validators.find((v) => v.mode === "evaluator") || validators[0];
    return pick ? pick.hotkey : null;
  }

  function selectValidator(hotkey) {
    if (hotkey === state.validator) return;
    state.validator = hotkey;
    const url = new URL(location.href);
    url.searchParams.set("validator", hotkey);
    history.replaceState(null, "", url);
    Object.assign(state.hotkeys, {page: 1});
    renderScoped();
    loadHotkeys();
  }

  // ---------- Queue ----------
  function renderQueue(data) {
    const target = $("#queue-body");
    if (!state.validator) { target.replaceChildren(empty("No validator selected", "Select a validator to see its queue.")); return; }
    const queue = data.queues ? data.queues[state.validator] : undefined;
    if (!Array.isArray(queue)) { target.replaceChildren(empty("No queue reported", "The backend reports no queue for this validator.")); return; }
    if (!queue.length) { target.replaceChildren(empty("Queue empty", "No challenges are waiting for this validator.")); return; }
    const perColdkey = new Map();
    for (const row of queue) perColdkey.set(row.coldkey, (perColdkey.get(row.coldkey) || 0) + 1);
    const rows = queue.map((row) => {
      const miner = el("span", null, "sn-stack");
      miner.append(who(row.uid, row.hotkey, "hotkey"));
      const cold = el("span", null, "sn-inline sn-small sn-muted");
      cold.append(el("span", "coldkey"), ident(row.coldkey, "coldkey"));
      miner.append(cold);
      const n = perColdkey.get(row.coldkey);
      if (known(row.coldkey) && n > 1) miner.append(el("small", `${n} hotkeys in this queue`, "sn-muted"));
      const position = finite(row.position) ? el("span", String(row.position), "sn-pos") : el("span", "—", "sn-muted");
      if (!finite(row.position)) position.title = "No position in the current window";
      return [position, miner, ident(row.model_id, "model id"), badge(row.status), reasonCell(row.reason), known(row.window_id) ? String(row.window_id) : "—", integer(row.block)];
    });
    const summary = queueSummary(queue);
    target.replaceChildren(summary.chips, table("Queue for the selected validator, in backend order", [["#", "sn-num"], ["Miner"], ["Model"], ["Status"], ["Why", "sn-wide sn-stackcell"], ["Window", "sn-num"], ["Block", "sn-num"]], rows,
      (i) => reasonInfo(queue[i].reason)?.tone === "bad" ? "sn-row-bad" : ""));
  }

  // ---------- Hotkeys (server paginated) ----------
  function renderCounts() {
    const v = currentValidator();
    $("#count-consumed").textContent = v ? integer(v.used_count) : "—";
    $("#count-reserved").textContent = v ? integer(v.reserved_count) : "—";
    for (const tab of document.querySelectorAll(".sn-tabs [role=tab]")) {
      const selected = tab.dataset.state === state.hotkeys.state;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected) $("#hotkeys-body").setAttribute("aria-labelledby", tab.id);
    }
  }

  async function loadHotkeys() {
    renderCounts();
    const target = $("#hotkeys-body");
    if (!state.validator) { target.replaceChildren(empty("No validator selected", "Hotkey lists are kept per validator.")); return; }
    const {state: which, q, page} = state.hotkeys;
    const seq = ++state.hotkeys.seq;
    target.setAttribute("aria-busy", "true");
    const params = new URLSearchParams({validator: state.validator, state: which, q, page: String(page)});
    try {
      const data = await read(`/api/hotkeys?${params}`);
      if (seq !== state.hotkeys.seq) return;
      renderHotkeys(data);
    } catch {
      if (seq === state.hotkeys.seq) failure(target, "Hotkey list", loadHotkeys);
    } finally {
      if (seq === state.hotkeys.seq) target.setAttribute("aria-busy", "false");
    }
  }

  function renderHotkeys(data) {
    const target = $("#hotkeys-body");
    const {state: which, q} = state.hotkeys;
    const rows = list(data.rows);
    const total = Number.isFinite(data.total) ? data.total : null;
    const size = Number.isFinite(data.page_size) && data.page_size > 0 ? data.page_size : 50;
    const page = Number.isFinite(data.page) ? data.page : state.hotkeys.page;
    const parts = [];
    if (!rows.length) {
      parts.push(q ? empty("No matches", `No ${which} hotkey matches “${q}” for this validator.`) : empty(`No ${which} hotkeys`, which === "consumed" ? "This validator has not consumed any hotkey yet." : "No hotkey is reserved for this validator."));
    } else {
      parts.push(table(`${which} hotkeys for the selected validator`, [["Hotkey"], ["Coldkey"], ["Model"], ["Status"], ["Window", "sn-num"], ["Reason", "sn-wide sn-stackcell"]],
        rows.map((r) => [who(r.uid, r.hotkey, "hotkey"), ident(r.coldkey, "coldkey"), ident(r.model_id, "model id"), badge(r.status), known(r.window_id) ? String(r.window_id) : "—", reasonCell(r.reason)])));
    }
    const pages = total == null ? null : Math.max(1, Math.ceil(total / size));
    const pager = el("nav", null, "sn-pager");
    pager.setAttribute("aria-label", "Hotkey pages");
    const prev = el("button", "← Previous", "sn-btn");
    prev.type = "button";
    prev.disabled = page <= 1;
    prev.onclick = () => { state.hotkeys.page = page - 1; loadHotkeys(); };
    const next = el("button", "Next →", "sn-btn");
    next.type = "button";
    next.disabled = pages == null ? rows.length < size : page >= pages;
    next.onclick = () => { state.hotkeys.page = page + 1; loadHotkeys(); };
    const first = (page - 1) * size + 1;
    const status = total == null ? `Page ${page}` : total === 0 ? "0 results" : `${integer(Math.min(first, total))}–${integer(Math.min(first + rows.length - 1, total))} of ${integer(total)} · page ${page} of ${pages}`;
    pager.append(prev, el("span", status), next);
    parts.push(pager);
    target.replaceChildren(...parts);
  }

  function setupHotkeyControls() {
    const tabs = [...document.querySelectorAll(".sn-tabs [role=tab]")];
    const choose = (tab) => {
      if (state.hotkeys.state === tab.dataset.state) return;
      state.hotkeys.state = tab.dataset.state;
      state.hotkeys.page = 1;
      tab.focus();
      loadHotkeys();
    };
    for (const tab of tabs) {
      tab.onclick = () => choose(tab);
      tab.onkeydown = (event) => {
        const i = tabs.indexOf(tab);
        const to = {ArrowRight: i + 1, ArrowLeft: i - 1 + tabs.length, Home: 0, End: tabs.length - 1}[event.key];
        if (to === undefined) return;
        event.preventDefault();
        choose(tabs[to % tabs.length]);
      };
    }
    const input = $("#hotkey-query");
    let timer = null;
    const search = () => {
      clearTimeout(timer);
      const q = input.value.trim();
      if (q === state.hotkeys.q) return;
      state.hotkeys.q = q;
      state.hotkeys.page = 1;
      loadHotkeys();
    };
    input.oninput = () => { clearTimeout(timer); timer = setTimeout(search, 350); };
    $("#hotkey-search").onsubmit = (event) => { event.preventDefault(); search(); };
  }

  // ---------- Results ----------
  function renderEvaluations(data) {
    const target = $("#evaluations-body");
    if (!state.validator) { target.replaceChildren(empty("No validator selected", "Select an evaluator to see its results.")); return; }
    const rows = list(data.evaluations).filter((e) => e.validator === state.validator);
    if (!rows.length) {
      const v = currentValidator();
      target.replaceChildren(empty("No results", v && v.mode === "follower" ? "This validator follows; it does not evaluate models." : "No result from this validator is published yet."));
      return;
    }
    // Group by window, newest first; rows keep backend order inside a window.
    const groups = new Map();
    for (const e of rows) {
      const key = finite(e.window_id) ? e.window_id : null;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(e);
    }
    const keys = [...groups.keys()].sort((a, b) => (b ?? -1) - (a ?? -1));
    const open = openWindowId(data);
    const kingModel = data.king?.model_id;
    const parts = keys.map((key) => {
      const items = groups.get(key);
      const section = el("section", null, "sn-rgroup");
      const head = el("div", null, "sn-rgroup-head");
      head.append(el("h3", key == null ? "Window unknown" : `Window ${key}`));
      if (items.some((e) => e.available)) head.append(el("span", "Closed · reports open", "sn-badge sn-good"));
      else if (key != null && key === open) head.append(el("span", "Open", "sn-badge sn-warn"), el("span", "Scores are published; clips and claims open after the window closes.", "sn-muted sn-small"));
      else head.append(el("span", "Report unavailable", "sn-badge sn-neutral"));
      const cells = items.map((e) => {
        const model = el("span", null, "sn-inline");
        model.append(ident(e.model_id, "model id"));
        if (known(kingModel) && e.model_id === kingModel) model.append(el("span", "Current king", "sn-tag"));
        const action = e.available ? reportButton(e, "Open report", "sn-btn sn-btn-small") : el("span", key != null && key === open ? "After close" : "Unavailable", "sn-muted sn-small");
        return [who(e.uid, e.hotkey, "hotkey"), model, evalStatus(e.status), meter(e.quality, e.quality_upper), meter(e.reward, e.reward_upper), known(e.report_hash) ? ident(e.report_hash, "report hash") : el("span", "Missing", "sn-unknown"), action];
      });
      section.append(head, table(`Results in ${key == null ? "an unknown window" : `window ${key}`} from the selected evaluator`, [["Miner"], ["Model"], ["Status"], ["Quality", "sn-num"], ["Reward", "sn-num"], ["Report hash"], ["", "sn-action"]], cells,
        (i) => known(kingModel) && items[i].model_id === kingModel ? "sn-kingrow" : ""));
      return section;
    });
    target.replaceChildren(...parts);
  }

  // ---------- Report dialog ----------
  function sameOrigin(url) {
    try { return new URL(url, location.href).origin === location.origin; } catch { return false; }
  }

  function responseBlock(label, response) {
    const details = el("details", null, "sn-response");
    details.append(el("summary", label));
    if (!known(response)) {
      details.append(el("p", "No response recorded.", "sn-muted"));
    } else {
      details.append(el("pre", typeof response === "string" ? response : JSON.stringify(response, null, 2)));
    }
    return details;
  }

  // Quality and reward, challenger against the paired king. Upper bounds stay marked; a missing king stays blank.
  function compare(caption, own, king, hasKing, upper = null) {
    const t = el("table", null, "sn-compare");
    t.append(el("caption", caption, "sn-visually-hidden"));
    const head = el("tr");
    for (const label of ["", "Quality", "Reward"]) {
      const th = el("th", label);
      th.scope = "col";
      head.append(th);
    }
    const thead = el("thead");
    thead.append(head);
    const tbody = el("tbody");
    const rows = [["Challenger", upper ? [null, upper.quality] : [own.quality], upper ? [null, upper.reward] : [own.reward]]];
    if (hasKing) rows.push(["King", [king.quality], [king.reward]]);
    for (const [label, q, r] of rows) {
      const tr = el("tr");
      const th = el("th", label);
      th.scope = "row";
      const tq = el("td");
      tq.append(meter(...q));
      const tr2 = el("td");
      tr2.append(meter(...r));
      tr.append(th, tq, tr2);
      tbody.append(tr);
    }
    t.append(thead, tbody);
    return t;
  }

  function versus(rows, hasKing) {
    const t = el("table", null, "sn-mini");
    t.append(el("caption", hasKing ? "Challenger compared with the king on the same clip" : "Challenger scores on this clip", "sn-visually-hidden"));
    const head = el("tr");
    for (const label of hasKing ? ["", "Challenger", "King"] : ["", "Challenger"]) {
      const th = el("th", label);
      th.scope = "col";
      head.append(th);
    }
    const thead = el("thead");
    thead.append(head);
    const tbody = el("tbody");
    for (const [label, a, b] of rows) {
      const tr = el("tr");
      const th = el("th", label);
      th.scope = "row";
      tr.append(th, el("td", a));
      if (hasKing) tr.append(el("td", b));
      tbody.append(tr);
    }
    t.append(thead, tbody);
    return t;
  }

  // Claims are the model's timestamped answer (clip-local seconds). Each time seeks the clip player.
  const MODALITIES = ["visual", "speech", "text", "sound"];
  const claimsOf = (response) => response && typeof response === "object" && Array.isArray(response.claims) ? response.claims : null;

  function claimRow(claim, video) {
    const li = el("li");
    const start = finite(claim?.start) ? claim.start : null;
    const end = finite(claim?.end) ? claim.end : null;
    if (start != null && end != null) Object.assign(li.dataset, {start: String(start), end: String(end)});
    const time = el("button", start != null && end != null ? `${start.toFixed(1)}–${end.toFixed(1)} s` : "Time ?", "sn-claim-time");
    time.type = "button";
    if (video && start != null) {
      time.setAttribute("aria-label", `Play the clip from ${start.toFixed(1)} seconds`);
      time.onclick = () => {
        video.currentTime = start;
        video.play().catch(() => {});
        video.scrollIntoView({block: "nearest"});
      };
    } else {
      time.disabled = true;
    }
    const modality = String(claim?.modality ?? "unknown");
    const chip = el("span", modality, MODALITIES.includes(modality) ? `sn-mod sn-mod-${modality}` : "sn-mod");
    const text = el("div", null, "sn-claim-text");
    text.append(el("strong", String(claim?.subject ?? "—")), el("span", String(claim?.description ?? "")));
    li.append(time, chip, text);
    return li;
  }

  function claimsColumn(label, response, video) {
    const column = el("div", null, "sn-claims-col");
    const claims = claimsOf(response);
    const head = el("p", null, "sn-claims-head");
    head.append(el("strong", label), el("span", claims ? plural(claims.length, "claim") : "", "sn-muted"));
    column.append(head);
    if (claims && claims.length) {
      const ol = el("ol", null, "sn-claims");
      for (const claim of claims) ol.append(claimRow(claim, video));
      column.append(ol);
    } else if (claims) {
      column.append(el("p", "The model returned no claims.", "sn-muted sn-claims-note"));
    } else if (known(response)) {
      column.append(el("p", "Not a valid claim list; shown exactly as returned.", "sn-muted sn-claims-note"), el("pre", typeof response === "string" ? response : JSON.stringify(response, null, 2), "sn-raw-response"));
    } else {
      column.append(el("p", "No response recorded.", "sn-muted sn-claims-note"));
    }
    return column;
  }

  // Claim intervals drawn against the clip length. Pointer shortcut only; the claim buttons are the accessible control.
  function timeline(lanes, duration, video) {
    const box = el("div", null, "sn-timeline");
    box.setAttribute("aria-hidden", "true");
    const heads = [];
    for (const [label, claims] of lanes) {
      const lane = el("div", null, "sn-lane");
      const track = el("div", null, "sn-track");
      for (const claim of list(claims)) {
        if (!finite(claim?.start) || !finite(claim?.end) || claim.end <= claim.start) continue;
        const kind = MODALITIES.includes(claim.modality) ? claim.modality : "other";
        const seg = el("i", null, `sn-seg sn-seg-${kind}`);
        const from = clamp(claim.start / duration);
        seg.style.left = `${from * 100}%`;
        seg.style.width = `${Math.max(0.6, (clamp(claim.end / duration) - from) * 100)}%`;
        track.append(seg);
      }
      const head = el("i", null, "sn-playhead");
      heads.push(head);
      track.append(head);
      if (video) {
        track.classList.add("sn-seekable");
        track.onclick = (event) => {
          const rect = track.getBoundingClientRect();
          video.currentTime = clamp((event.clientX - rect.left) / rect.width) * duration;
          video.play().catch(() => {});
        };
      }
      lane.append(el("span", label, "sn-lane-label"), track);
      box.append(lane);
    }
    const scale = el("div", null, "sn-lane sn-scale");
    const marks = el("span", null, "sn-scale-marks");
    marks.append(el("span", "0 s"), el("span", `${decimal(duration, 1)} s`));
    scale.append(el("span"), marks);
    box.append(scale);
    return {box, move: (t) => { for (const head of heads) head.style.left = `${clamp(t / duration) * 100}%`; }};
  }

  function clipCard(clip, index) {
    const article = el("article", null, "sn-clip");
    const head = el("div", null, "sn-clip-head");
    const range = known(clip.start) && known(clip.duration) ? `source ${decimal(clip.start, 1)} s → ${decimal(clip.start + clip.duration, 1)} s · ${decimal(clip.duration, 1)} s clip` : "Source range unknown";
    const name = el("strong", `Clip ${index + 1} `);
    name.append(ident(clip.id, "clip id"));
    head.append(name, el("span", range, "sn-muted"));
    article.append(head);
    const body = el("div", null, "sn-clip-body");
    const media = el("div", null, "sn-clip-media");
    const k = clip.king;
    let video = null;
    if (known(clip.video_url) && sameOrigin(clip.video_url)) {
      video = el("video");
      video.controls = true;
      video.preload = "metadata";
      video.muted = true;
      video.playsInline = true;
      // video_url already serves the cut clip; start/duration describe the source and are informational only.
      video.src = new URL(clip.video_url, location.href).href;
      video.setAttribute("aria-label", `Clip ${index + 1} video`);
      media.append(video);
    } else {
      media.append(el("p", "Clip media not available.", "sn-muted sn-no-media"));
    }
    const lanes = [["Challenger", claimsOf(clip.response)]];
    if (k) lanes.push(["King", claimsOf(k.response)]);
    const line = finite(clip.duration) && clip.duration > 0 ? timeline(lanes, clip.duration, video) : null;
    if (line) media.append(line.box);
    if (video) {
      // Mark the claims (challenger and king) whose interval covers the playhead.
      video.addEventListener("timeupdate", () => {
        const t = video.currentTime;
        if (line) line.move(t);
        for (const li of article.querySelectorAll(".sn-claims li[data-start]")) li.classList.toggle("sn-claim-now", t >= Number(li.dataset.start) && t < Number(li.dataset.end));
      });
    }
    media.append(versus([
      ["Quality", decimal(clip.quality), decimal(k?.quality)],
      ["Latency", known(clip.latency_s) ? `${decimal(clip.latency_s, 2)} s` : "—", known(k?.latency_s) ? `${decimal(k.latency_s, 2)} s` : "—"],
      ["Time score", decimal(clip.time_score), decimal(k?.time_score)],
      ["Reward", decimal(clip.reward), decimal(k?.reward)],
    ], Boolean(k)));
    const claims = el("div", null, k ? "sn-claims-grid" : "sn-claims-grid sn-claims-single");
    claims.append(claimsColumn("Challenger claims", clip.response, video));
    if (k) claims.append(claimsColumn("King claims", k.response, video));
    body.append(media, claims);
    article.append(body);
    return article;
  }

  function sideCard(eyebrow, identity, model, numbers) {
    const box = el("div", null, "sn-side");
    box.append(el("p", eyebrow, "sn-eyebrow"), facts([["Miner", identity], ["Model", model]]));
    if (numbers) box.append(numbers);
    return box;
  }

  function sideNumbers(quality, reward, upper) {
    const dl = el("dl", null, "sn-kr-nums");
    for (const [label, value, bound] of [["Quality", quality, upper?.quality], ["Reward", reward, upper?.reward]]) {
      const row = el("div");
      const dd = el("dd");
      dd.append(upper ? meter(null, bound) : meter(value, null));
      row.append(el("dt", label), dd);
      dl.append(row);
    }
    return dl;
  }

  function renderDetail(detail, evaluation) {
    const body = $("#detail-body");
    if (!detail || detail.available !== true) {
      body.replaceChildren(empty("Report not available", detail?.reason || "The backend did not provide a reason."));
      return;
    }
    const hasKing = Boolean(detail.opponent);
    const t = detail.total || {};
    const early = detail.evaluation_status === "early_stop";
    const upper = early ? {quality: t.quality_upper ?? detail.early_stop?.quality_upper, reward: t.reward_upper ?? detail.early_stop?.reward_upper} : null;
    const parts = [];
    const pair = el("div", null, "sn-pair");
    const uid = finite(detail.uid) ? detail.uid : evaluation?.hotkey === detail.hotkey ? evaluation?.uid : null;
    pair.append(sideCard(early ? "CHALLENGER · UPPER BOUND" : "CHALLENGER", who(uid, detail.hotkey, "challenger hotkey"), ident(detail.model_id, "challenger model id"), sideNumbers(t.quality, t.reward, upper)));
    if (hasKing) {
      pair.append(sideCard("PAIRED KING · SAME BATCH", who(detail.opponent.uid, detail.opponent.hotkey, "king hotkey"), ident(detail.opponent.model_id, "king model id"), sideNumbers(t.king_quality, t.king_reward, null)));
    } else {
      const none = el("div", null, "sn-side sn-side-empty");
      none.append(el("p", "PAIRED KING", "sn-eyebrow"), el("p", "No king was paired with this evaluation.", "sn-muted"));
      pair.append(none);
    }
    parts.push(pair);
    if (early) {
      const stop = detail.early_stop || {};
      const box = el("div", null, "sn-warnings");
      box.append(el("strong", `Early stop · ${stop.observed_videos ?? "?"} of ${stop.planned_videos ?? "?"} videos evaluated`));
      const items = el("ul");
      const method = stop.method === "finite_batch_90"
        ? `Statistical futility check at ${known(stop.confidence) ? Math.round(stop.confidence * 100) : "?"}% confidence for this fixed batch only, not for the model in general.`
        : stop.method === "best_possible_completion" ? "Even a perfect score on the remaining videos could not win." : `Method: ${stop.method ?? "unknown"}.`;
      for (const text of [method, "The challenger total is an upper bound, not a measured score. A partial result never crowns a model.", "Evaluated clips below are measured; unevaluated videos are not shown."]) items.append(el("li", text));
      box.append(items);
      parts.push(box);
    }
    const videos = list(detail.videos);
    const clipCount = videos.reduce((n, video) => n + list(video.clips).length, 0);
    const meta = el("p", null, "sn-note sn-report-meta");
    meta.append(el("span", `${plural(videos.length, "video")} · ${plural(clipCount, "clip")} · report `), known(detail.report_hash) ? ident(detail.report_hash, "report hash") : el("span", "hash missing", "sn-unknown"), el("span", " · reward = quality × (0.8 + 0.2 × time_score), computed by the evaluator. Select a claim time to play the clip from that moment."));
    parts.push(meta);
    if (!videos.length) parts.push(empty("No clips in report", "The report lists no sampled videos."));
    const sections = videos.map((video, i) => {
      const section = el("section", null, "sn-video");
      const head = el("div", null, "sn-video-head");
      const title = el("h3", `Video ${i + 1}`);
      title.tabIndex = -1;
      const id = el("span", null, "sn-video-id");
      id.append(ident(video.id, "video id"));
      const label = el("div");
      label.append(title, id);
      head.append(label, compare(`Video ${i + 1} average`, video, {quality: video.king_quality, reward: video.king_reward}, hasKing));
      section.append(head);
      list(video.clips).forEach((clip, c) => section.append(clipCard(clip, c)));
      return {section, title};
    });
    if (sections.length > 1) {
      const nav = el("nav", null, "sn-jump");
      nav.setAttribute("aria-label", "Videos in this report");
      sections.forEach(({section, title}, i) => {
        const button = el("button", `Video ${i + 1}`, "sn-btn sn-btn-small");
        button.type = "button";
        const reward = videos[i].reward;
        if (finite(reward)) button.append(el("span", decimal(reward), "sn-muted"));
        button.onclick = () => { section.scrollIntoView({block: "start"}); title.focus({preventScroll: true}); };
        nav.append(button);
      });
      parts.push(nav);
    }
    parts.push(...sections.map((s) => s.section));
    body.replaceChildren(...parts);
  }

  let detailTrigger = null;
  let detailSeq = 0;
  async function openDetail(evaluation, trigger) {
    const dialog = $("#detail");
    detailTrigger = trigger;
    $("#detail-context").textContent = `WINDOW ${evaluation.window_id ?? "?"} · EVALUATOR ${short(String(evaluation.validator ?? ""))}`;
    $("#detail-body").replaceChildren(el("p", "Loading report…", "sn-loading"));
    if (!dialog.open) dialog.showModal();
    const seq = ++detailSeq;
    const path = ["/api/evaluations", evaluation.validator, evaluation.window_id, evaluation.model_id].map((part, i) => i ? encodeURIComponent(String(part)) : part).join("/");
    try {
      const detail = await read(path);
      if (seq === detailSeq) renderDetail(detail, evaluation);
    } catch {
      if (seq === detailSeq) failure($("#detail-body"), "Evaluation report", () => openDetail(evaluation, trigger));
    }
  }

  function setupDialog() {
    const dialog = $("#detail");
    // The sticky video jump bar sits under the sticky dialog header, whose height depends on wrapping.
    const head = dialog.querySelector(".sn-dialog-head");
    new ResizeObserver(() => dialog.style.setProperty("--dialog-head-h", `${head.offsetHeight}px`)).observe(head);
    $("#detail-close").onclick = () => dialog.close();
    dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
    dialog.addEventListener("close", () => {
      detailSeq++;
      for (const video of dialog.querySelectorAll("video")) video.pause();
      if (detailTrigger && detailTrigger.isConnected) detailTrigger.focus();
    });
  }

  // ---------- Load cycle ----------
  function renderScoped() {
    if (!state.data) return;
    renderValidators(state.data);
    renderScope(state.data);
    renderQueue(state.data);
    renderCounts();
    renderEvaluations(state.data);
  }

  async function load() {
    if (state.loading) return;
    state.loading = true;
    const button = $("#refresh");
    button.disabled = true;
    try {
      const data = await read("/api/subnet");
      if (data.schema_version !== "witness-dashboard-2") throw new Error("schema");
      const previous = state.validator;
      state.data = data;
      state.fetchedAt = Date.now();
      state.failed = false;
      state.validator = defaultValidator(list(data.validators));
      renderChain(data);
      renderErrors(data.errors);
      renderOverview(data);
      renderScoped();
      if (previous !== state.validator || !$("#hotkeys-body .sn-table, #hotkeys-body .sn-empty")) loadHotkeys();
    } catch {
      state.failed = true;
      if (!state.data) {
        for (const id of ["#validators-body", "#queue-body", "#hotkeys-body", "#evaluations-body"]) $(id).replaceChildren(el("p", "Unavailable until the subnet state loads.", "sn-loading"));
        const select = $("#validator-select");
        select.replaceChildren(el("option", "Unavailable"));
        select.disabled = true;
        failure($("#overview-body"), "Subnet state", load);
      }
    } finally {
      state.loading = false;
      button.disabled = false;
      renderUpdated();
      renderStale();
    }
  }

  // The sticky validator bar sits under the sticky topbar, whose height grows when chain facts wrap.
  const topbar = $(".sn-topbar");
  new ResizeObserver(() => document.documentElement.style.setProperty("--topbar-h", `${topbar.offsetHeight}px`)).observe(topbar);

  $("#refresh").onclick = load;
  $("#validator-select").onchange = (event) => selectValidator(event.target.value);
  setupHotkeyControls();
  setupDialog();
  load();
  // Finalized state changes at block cadence; refresh quietly while the tab is visible.
  setInterval(() => { if (!document.hidden && !$("#detail").open) load(); }, 60000);
  setInterval(() => {
    if (document.hidden) return;
    for (const span of document.querySelectorAll("[data-since]")) tick(span);
    renderUpdated();
  }, 1000);
})();
