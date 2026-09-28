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
  const integer = (n) => known(n) && Number.isFinite(n) ? new Intl.NumberFormat("en-US").format(n) : "Unknown";
  const decimal = (n, digits = 3) => known(n) && Number.isFinite(n) ? n.toFixed(digits) : "—";
  const stake = (n) => known(n) && Number.isFinite(n) ? new Intl.NumberFormat("en-US", {maximumFractionDigits: 2}).format(n) : "Unknown";
  const short = (s) => s.length > 18 ? `${s.slice(0, 8)}…${s.slice(-6)}` : s;
  const finite = (n) => typeof n === "number" && Number.isFinite(n);
  const list = (v) => Array.isArray(v) ? v : [];
  const pct = (f) => finite(f) ? `${Math.round(f * 1000) / 10}%` : "?";
  const clock = (ms) => new Date(ms).toLocaleTimeString(undefined, {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  const age = (s) => s < 60 ? `${s} s` : s < 3600 ? `${Math.floor(s / 60)} min` : `${Math.floor(s / 3600)} h`;
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

  const TONES = {
    good: ["fresh", "applied", "consumed", "evaluated", "done", "complete", "completed", "closed", "success", "finalized", "included", "won"],
    warn: ["stale", "reserved", "queued", "pending", "waiting", "running", "retry", "open", "submitted", "evaluating"],
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
  function score(value, upper) {
    if (known(value) || !known(upper)) return decimal(value);
    const box = el("span", null, "sn-stack");
    box.title = "Upper bound from an early stop, not a measured score";
    box.append(el("span", `≤ ${decimal(upper)}`), el("small", "upper bound", "sn-muted"));
    return box;
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
    add("Fresh", validators.length ? el("span", `${fresh} of ${validators.length} reporting`, fresh === validators.length ? "sn-ok" : "sn-warn") : unknown("None reporting"));
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

  function renderErrors(errors) {
    const target = $("#errors");
    target.replaceChildren();
    if (!Array.isArray(errors) || !errors.length) return;
    const box = el("div", null, "sn-warnings");
    box.append(el("strong", `${errors.length} source${errors.length === 1 ? "" : "s"} reported an error`));
    const list = el("ul");
    for (const item of errors) {
      const li = el("li");
      li.append(el("code", item.source ?? "unknown source"), el("span", ` ${item.error ?? "unknown error"}`));
      list.append(li);
    }
    box.append(list);
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

  function renderCrown(king, intended, authority) {
    const title = el("div", null, "sn-card-title");
    const body = [];
    if (!authority && !king) {
      // Without the authoritative chain projection, "no king" would be a guess.
      title.append(unknown("King unknown"));
      body.push(el("p", "The authoritative chain source is unavailable. King, burn and weights stay unknown until it answers.", "sn-lede"));
      return card("KING", title, body, el("span", "Unknown", "sn-badge sn-neutral"), "sn-crown");
    }
    if (king) {
      title.append(ident(king.hotkey, "king hotkey"));
      body.push(facts([["Coldkey", ident(king.coldkey, "king coldkey")], ["Model", ident(king.model_id, "king model id")], ["UID", known(intended?.king_uid) ? String(intended.king_uid) : unknown()]]));
    } else {
      title.append(el("span", "No king yet"));
      const burn = finite(intended?.burn_fraction) ? `${pct(intended.burn_fraction)} of the weight burns` : "Weight burns";
      const uid = known(intended?.burn_uid) ? `UID ${intended.burn_uid}` : "the burn UID";
      body.push(el("p", `${burn} to ${uid} until the chain crowns a first king. The first king comes from a bootstrap pair, so the subnet waits for two valid submissions evaluated on the same batch.`, "sn-lede"));
    }
    body.push(el("p", "WEIGHT SPLIT · INTENDED", "sn-eyebrow sn-sub-eyebrow"), splitBar(intended));
    body.push(el("p", "The king is read from finalized chain consensus only. With a king, 30% goes to the king and 70% burns; without one, 100% burns.", "sn-note"));
    const status = king ? el("span", "Crowned", "sn-badge sn-good") : el("span", "No king · burning", "sn-badge sn-warn");
    return card("KING", title, body, status, "sn-crown");
  }

  function renderWeights(weights) {
    const {decision, intended, submitted, applied} = weights;
    const steps = el("ol", null, "sn-steps");
    const step = (label, item, hint, fill, extra) => {
      const li = el("li", null, item ? "" : "sn-step-missing");
      const top = el("div", null, "sn-step-top");
      top.append(el("strong", label));
      if (extra) top.append(...extra.filter(Boolean));
      li.append(top);
      if (item) fill(li); else li.append(unknown(label === "Applied" ? "Not observed on finalized chain" : "None reported"));
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
    step("Submitted", submitted, "set_weights extrinsic sent; not proof that weights applied", (li) => {
      if (Array.isArray(submitted.uids)) li.append(vector(submitted.uids, submitted.weights));
      li.append(el("small", `Block ${integer(submitted.block)}`));
      if (known(submitted.receipt)) li.append(responseBlock("Receipt", submitted.receipt));
    }, [submitted ? badge(submitted.status) : null,
      submitted && intended && Array.isArray(submitted.uids) && !sameVector(submitted, intended) ? el("span", "differs from intended", "sn-badge sn-warn") : null]);
    step("Applied", applied, "Observed in finalized chain state; commit/reveal can delay it after submission", (li) => {
      if (known(applied.hotkey)) li.append(ident(applied.hotkey, "applied king hotkey"));
      li.append(vector(applied.uids, applied.weights), el("small", `Block ${integer(applied.block)}`));
    }, [applied && intended ? (sameVector(applied, intended) ? el("span", "matches intended", "sn-badge sn-good") : el("span", "differs from intended", "sn-badge sn-warn")) : null]);
    return card("WEIGHTS · DECISION → APPLIED", null, [steps], null, "sn-card-wide");
  }

  function modelRole(modelId, king) {
    if (!known(modelId)) return "No model";
    if (king && king.model_id === modelId) return "King baseline";
    return king ? "Challenger" : "Bootstrap candidate";
  }

  function liveTile(v, data) {
    const p = v.progress;
    const tile = el("div", null, "sn-live-tile");
    const top = el("div", null, "sn-live-top");
    top.append(ident(v.hotkey, "validator hotkey"), badge(v.freshness));
    tile.append(top);
    const queue = data.queues ? data.queues[v.hotkey] : undefined;
    const queueText = Array.isArray(queue) ? `${integer(queue.length)} in queue` : "Queue unknown";
    if (!p) {
      tile.append(el("div", null, "sn-live-stage"));
      tile.lastChild.append(unknown("No progress reported"));
      tile.append(el("p", "State unknown until this validator publishes telemetry.", "sn-muted sn-small"));
    } else {
      const stage = el("div", null, "sn-live-stage");
      stage.append(stageBadge(p.stage), el("span", known(p.window_id) ? `Window ${p.window_id}` : "Window unknown", "sn-muted"));
      const model = el("div", null, "sn-live-model");
      model.append(el("span", modelRole(p.model_id, data.king), "sn-role"));
      if (known(p.model_id)) model.append(ident(p.model_id, "model id"));
      tile.append(stage, model, clipProgress(p.completed_clips, p.total_clips));
    }
    const foot = el("div", null, "sn-live-foot");
    foot.append(p ? since(p.updated_unix, "Updated ") : el("span", v.mode ?? "unknown mode"), el("span", queueText));
    tile.append(foot);
    return tile;
  }

  function renderLive(data) {
    const validators = list(data.validators).filter((v) => v.mode === "evaluator" || known(v.progress));
    const grid = el("div", null, "sn-live-grid");
    if (!validators.length) grid.append(empty("No evaluator telemetry", "No validator publishes progress to this dashboard. Telemetry is opt-in, so this says nothing about evaluations or commitments on chain."));
    for (const v of validators) grid.append(liveTile(v, data));
    const note = el("p", "Signed validator telemetry, shown for display only. It never decides the king or the weights. Open-window clips stay hidden until the window closes.", "sn-note");
    return card("EVALUATION NOW · TELEMETRY", null, [grid, note], null, "sn-card-wide");
  }

  function renderOverview(data) {
    const target = $("#overview-body");
    target.replaceChildren();
    const weights = data.weights || {};
    target.append(renderCrown(data.king, weights.intended, known(data.block)));

    const w = data.window;
    const windowTitle = el("div", null, "sn-card-title");
    windowTitle.append(el("span", w ? `Window ${w.id}` : "No window reported"));
    const windowBody = w ? [facts([
      ["Epochs", known(w.start_epoch) ? `${w.start_epoch} – ${known(w.end_epoch) ? w.end_epoch : "?"}` : "Unknown"],
      ["Start block", integer(w.start_block)],
      ["End block", known(w.end_block) ? integer(w.end_block) : el("span", "Pending finalization", "sn-unknown")],
    ])] : [el("p", "No evaluation window is reported.", "sn-note")];
    if (w && w.state === "open") windowBody.push(el("p", "Clip detail for this window is hidden until it closes.", "sn-note"));
    target.append(card("WINDOW · 2 EPOCHS", windowTitle, windowBody, w ? badge(w.state) : null));
    target.append(renderWeights(weights), renderLive(data));
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

  function renderValidators(data) {
    const target = $("#validators-body");
    const validators = Array.isArray(data.validators) ? data.validators : [];
    if (!validators.length) {
      target.replaceChildren(empty("No validator reporting", "No validator publishes its status to this dashboard. Reporting is opt-in: registered validators may still hold chain commitments that are not listed here."));
      return;
    }
    const rows = validators.map((v) => {
      const select = el("button", v.hotkey === state.validator ? "Selected" : "Inspect", "sn-link");
      select.type = "button";
      select.setAttribute("aria-pressed", String(v.hotkey === state.validator));
      select.onclick = () => { selectValidator(v.hotkey); $("#queue").scrollIntoView({block: "start"}); };
      return [ident(v.hotkey, "validator hotkey"), badge(v.mode), stake(v.stake), badge(v.freshness), integer(v.block), known(v.commitment_block) ? integer(v.commitment_block) : el("span", "None", "sn-unknown"), progressCell(v.progress), integer(v.used_count), integer(v.reserved_count), select];
    });
    target.replaceChildren(table("Validators", [["Hotkey"], ["Mode"], ["Stake", "sn-num"], ["Freshness"], ["Last block", "sn-num"], ["Commitment block", "sn-num"], ["Progress"], ["Consumed", "sn-num"], ["Reserved", "sn-num"], ["", "sn-action"]], rows,
      (i) => validators[i].hotkey === state.validator ? "sn-selected" : ""));
  }

  function renderScope(data) {
    const select = $("#validator-select");
    const validators = Array.isArray(data.validators) ? data.validators : [];
    select.replaceChildren();
    for (const v of validators) {
      const option = el("option", `${short(String(v.hotkey))} · ${v.mode ?? "unknown"}`);
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
      const cold = el("span", null, "sn-stack");
      cold.append(ident(row.coldkey, "coldkey"));
      const n = perColdkey.get(row.coldkey);
      if (known(row.coldkey) && n > 1) cold.append(el("small", `${n} hotkeys in queue`, "sn-muted"));
      return [known(row.position) ? String(row.position) : "—", ident(row.hotkey, "hotkey"), cold, ident(row.model_id, "model id"), integer(row.block), badge(row.status), badge(row.usage), known(row.window_id) ? String(row.window_id) : "—", row.reason ? el("span", row.reason, "sn-reason") : el("span", "—", "sn-muted")];
    });
    target.replaceChildren(table("Fair queue for the selected validator, in backend order", [["#", "sn-num"], ["Hotkey"], ["Coldkey"], ["Model"], ["Block", "sn-num"], ["Status"], ["Usage"], ["Window", "sn-num"], ["Reason", "sn-wide"]], rows));
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
    const rows = Array.isArray(data.rows) ? data.rows : [];
    const total = Number.isFinite(data.total) ? data.total : null;
    const size = Number.isFinite(data.page_size) && data.page_size > 0 ? data.page_size : 50;
    const page = Number.isFinite(data.page) ? data.page : state.hotkeys.page;
    const parts = [];
    if (!rows.length) {
      parts.push(q ? empty("No matches", `No ${which} hotkey matches “${q}” for this validator.`) : empty(`No ${which} hotkeys`, which === "consumed" ? "This validator has not consumed any hotkey yet." : "No hotkey is reserved for this validator."));
    } else {
      parts.push(table(`${which} hotkeys for the selected validator`, [["Hotkey", "sn-wide"], ["Coldkey", "sn-wide"], ["Model"], ["Status"], ["Window", "sn-num"], ["Reason"]],
        rows.map((r) => [ident(r.hotkey, "hotkey", true), ident(r.coldkey, "coldkey", true), ident(r.model_id, "model id"), badge(r.status), known(r.window_id) ? String(r.window_id) : "—", r.reason ? el("span", r.reason, "sn-reason") : el("span", "—", "sn-muted")])));
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
        if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
        event.preventDefault();
        choose(tabs[(tabs.indexOf(tab) + 1) % tabs.length]);
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

  // ---------- Evaluations ----------
  function renderEvaluations(data) {
    const target = $("#evaluations-body");
    if (!state.validator) { target.replaceChildren(empty("No validator selected", "Select an evaluator to see its evaluations.")); return; }
    const all = Array.isArray(data.evaluations) ? data.evaluations : [];
    const list = all.filter((e) => e.validator === state.validator);
    if (!list.length) {
      const v = currentValidator();
      target.replaceChildren(empty("No evaluations", v && v.mode === "follower" ? "This validator follows; it does not evaluate models." : "No evaluation from this validator is published yet."));
      return;
    }
    const openWindow = data.window && data.window.state === "open" ? data.window.id : null;
    const rows = list.map((e) => {
      let action;
      if (e.available) {
        action = el("button", "Open detail", "sn-link");
      } else {
        action = el("button", e.window_id === openWindow ? "After close" : "Unavailable", "sn-link sn-link-muted");
      }
      action.type = "button";
      action.dataset.action = "detail";
      action.setAttribute("aria-haspopup", "dialog");
      action.onclick = () => openDetail(e, action);
      return [known(e.window_id) ? String(e.window_id) : "—", ident(e.hotkey, "hotkey"), ident(e.model_id, "model id"), evalStatus(e.status), score(e.quality, e.quality_upper), score(e.reward, e.reward_upper), known(e.report_hash) ? ident(e.report_hash, "report hash") : el("span", "Missing", "sn-unknown"), action];
    });
    target.replaceChildren(table("Evaluations published by the selected validator", [["Window", "sn-num"], ["Hotkey"], ["Model"], ["Status"], ["Quality", "sn-num"], ["Reward", "sn-num"], ["Report"], ["", "sn-action"]], rows));
  }

  // ---------- Detail dialog ----------
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

  function versus(rows, hasKing) {
    const t = el("table", null, "sn-mini");
    t.append(el("caption", "Challenger compared with the king on the same clip", "sn-visually-hidden"));
    const head = el("tr");
    for (const label of ["", "Challenger", "King"]) {
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
      tr.append(th, el("td", a), el("td", hasKing ? b : "—"));
      tbody.append(tr);
    }
    t.append(thead, tbody);
    return t;
  }

  // Claims are the model's timestamped answer (clip-local seconds). Each time seeks the clip player.
  const MODALITIES = ["visual", "speech", "text", "sound"];
  function claimRow(claim, video) {
    const li = el("li");
    const start = finite(claim?.start) ? claim.start : null;
    const end = finite(claim?.end) ? claim.end : null;
    if (start != null && end != null) Object.assign(li.dataset, {start: String(start), end: String(end)});
    const time = el("button", start != null && end != null ? `${start.toFixed(1)}–${end.toFixed(1)} s` : "Time ?", "sn-claim-time");
    time.type = "button";
    if (video && start != null) {
      time.setAttribute("aria-label", `Play the clip from ${start.toFixed(1)} seconds`);
      time.onclick = () => { video.currentTime = start; video.play().catch(() => {}); };
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

  function claimsBlock(label, response, video, open) {
    const details = el("details", null, "sn-response");
    details.open = open;
    const claims = response && typeof response === "object" && Array.isArray(response.claims) ? response.claims : null;
    details.append(el("summary", claims ? `${label} · ${claims.length} claim${claims.length === 1 ? "" : "s"}` : label));
    if (claims && claims.length) {
      const ol = el("ol", null, "sn-claims");
      for (const claim of claims) ol.append(claimRow(claim, video));
      details.append(ol);
    } else if (claims) {
      details.append(el("p", "The model returned no claims.", "sn-muted"));
    } else if (known(response)) {
      details.append(el("p", "Not a valid claim list; shown exactly as returned.", "sn-muted"), el("pre", typeof response === "string" ? response : JSON.stringify(response, null, 2)));
    } else {
      details.append(el("p", "No response recorded.", "sn-muted"));
    }
    return details;
  }

  function clipCard(clip) {
    const article = el("article", null, "sn-clip");
    const head = el("div", null, "sn-clip-head");
    const range = known(clip.start) && known(clip.duration) ? `source ${decimal(clip.start, 1)} s → ${decimal(clip.start + clip.duration, 1)} s · ${decimal(clip.duration, 1)} s clip` : "Source range unknown";
    const name = el("strong", "Clip ");
    name.append(el("code", short(String(clip.id ?? "?"))));
    head.append(name, el("span", range, "sn-muted"));
    article.append(head);
    const body = el("div", null, "sn-clip-body");
    const media = el("div", null, "sn-clip-media");
    let video = null;
    if (known(clip.video_url) && sameOrigin(clip.video_url)) {
      video = el("video");
      video.controls = true;
      video.preload = "metadata";
      video.muted = true;
      video.playsInline = true;
      // video_url already serves the cut clip; start/duration describe the source and are informational only.
      video.src = new URL(clip.video_url, location.href).href;
      video.setAttribute("aria-label", `Clip ${clip.id ?? ""} video`);
      // Mark the claims (challenger and king) whose interval covers the playhead.
      video.addEventListener("timeupdate", () => {
        const t = video.currentTime;
        for (const li of article.querySelectorAll(".sn-claims li[data-start]")) li.classList.toggle("sn-claim-now", t >= Number(li.dataset.start) && t < Number(li.dataset.end));
      });
      media.append(video);
    } else {
      media.append(el("p", "Clip media not available.", "sn-muted"));
    }
    const k = clip.king;
    const stats = el("div", null, "sn-clip-stats");
    stats.append(versus([
      ["Quality", decimal(clip.quality), decimal(k?.quality)],
      ["Latency", known(clip.latency_s) ? `${decimal(clip.latency_s, 2)} s` : "—", known(k?.latency_s) ? `${decimal(k.latency_s, 2)} s` : "—"],
      ["Time score", decimal(clip.time_score), decimal(k?.time_score)],
      ["Reward", decimal(clip.reward), decimal(k?.reward)],
    ], Boolean(k)));
    stats.append(claimsBlock("Challenger claims", clip.response, video, true));
    if (k) stats.append(claimsBlock("King claims", k.response, video, false));
    body.append(media, stats);
    article.append(body);
    return article;
  }

  function totals(label, quality, reward, kingQuality, kingReward, hasKing, upper = null) {
    const box = el("div", null, "sn-totals");
    box.append(el("p", label, "sn-eyebrow"));
    const grid = el("dl");
    const own = upper ? [["Quality upper bound", upper.quality, "≤ "], ["Reward upper bound", upper.reward, "≤ "]] : [["Quality", quality, ""], ["Reward", reward, ""]];
    for (const [name, value, prefix = ""] of [...own, ["King quality", hasKing ? kingQuality : null], ["King reward", hasKing ? kingReward : null]]) {
      const row = el("div");
      row.append(el("dt", name), el("dd", known(value) ? prefix + decimal(value) : decimal(value)));
      grid.append(row);
    }
    box.append(grid);
    return box;
  }

  function renderDetail(detail) {
    const body = $("#detail-body");
    if (!detail || detail.available !== true) {
      body.replaceChildren(empty("Detail not available", detail?.reason || "The backend did not provide a reason."));
      return;
    }
    const hasKing = Boolean(detail.opponent);
    const parts = [];
    const pair = el("div", null, "sn-pair");
    const challenger = el("div");
    challenger.append(el("p", "CHALLENGER", "sn-eyebrow"), facts([["Hotkey", ident(detail.hotkey, "challenger hotkey")], ["Model", ident(detail.model_id, "challenger model id")]]));
    const king = el("div");
    king.append(el("p", "PAIRED KING", "sn-eyebrow"), hasKing ? facts([["Hotkey", ident(detail.opponent.hotkey, "king hotkey")], ["Model", ident(detail.opponent.model_id, "king model id")]]) : el("p", "No king was paired with this evaluation.", "sn-muted"));
    pair.append(challenger, king);
    parts.push(pair);
    const t = detail.total || {};
    const early = detail.evaluation_status === "early_stop";
    if (early) {
      const stop = detail.early_stop || {};
      const box = el("div", null, "sn-warnings");
      box.append(el("strong", `Early stop · ${stop.observed_videos ?? "?"} of ${stop.planned_videos ?? "?"} videos evaluated`));
      const list = el("ul");
      const method = stop.method === "finite_batch_90"
        ? `Statistical futility check at ${known(stop.confidence) ? Math.round(stop.confidence * 100) : "?"}% confidence for this fixed batch only, not for the model in general.`
        : stop.method === "best_possible_completion" ? "Even a perfect score on the remaining videos could not win." : `Method: ${stop.method ?? "unknown"}.`;
      for (const text of [method, "The challenger total is an upper bound, not a measured score. A partial result never crowns a model.", "Evaluated clips below are measured; unevaluated videos are not shown."]) list.append(el("li", text));
      box.append(list);
      parts.push(box);
    }
    parts.push(totals(early ? "TOTAL · UPPER BOUND" : "TOTAL", t.quality, t.reward, t.king_quality, t.king_reward, hasKing,
      early ? {quality: t.quality_upper ?? detail.early_stop?.quality_upper, reward: t.reward_upper ?? detail.early_stop?.reward_upper} : null));
    const videos = Array.isArray(detail.videos) ? detail.videos : [];
    const clipCount = videos.reduce((n, video) => n + list(video.clips).length, 0);
    const meta = el("p", null, "sn-note");
    meta.append(el("span", `${videos.length} video${videos.length === 1 ? "" : "s"} · ${clipCount} clip${clipCount === 1 ? "" : "s"} in this report · Report `), known(detail.report_hash) ? ident(detail.report_hash, "report hash") : el("span", "hash missing", "sn-unknown"), el("span", " · reward = quality × (0.8 + 0.2 × time_score), computed by the evaluator. Claim times play the clip from that moment."));
    parts.push(meta);
    if (!videos.length) parts.push(empty("No clips in report", "The report lists no sampled videos."));
    for (const video of videos) {
      const section = el("section", null, "sn-video");
      const head = el("div", null, "sn-video-head");
      head.append(el("h3", `Video ${video.id ?? "?"}`));
      head.append(totals("VIDEO AVERAGE", video.quality, video.reward, video.king_quality, video.king_reward, hasKing));
      section.append(head);
      for (const clip of Array.isArray(video.clips) ? video.clips : []) section.append(clipCard(clip));
      parts.push(section);
    }
    body.replaceChildren(...parts);
  }

  let detailTrigger = null;
  let detailSeq = 0;
  async function openDetail(evaluation, trigger) {
    const dialog = $("#detail");
    detailTrigger = trigger;
    $("#detail-context").textContent = `WINDOW ${evaluation.window_id ?? "?"} · VALIDATOR ${short(String(evaluation.validator ?? ""))}`;
    $("#detail-body").replaceChildren(el("p", "Loading evaluation detail…", "sn-loading"));
    if (!dialog.open) dialog.showModal();
    const seq = ++detailSeq;
    const path = ["/api/evaluations", evaluation.validator, evaluation.window_id, evaluation.model_id].map((part, i) => i ? encodeURIComponent(String(part)) : part).join("/");
    try {
      const detail = await read(path);
      if (seq === detailSeq) renderDetail(detail);
    } catch {
      if (seq === detailSeq) failure($("#detail-body"), "Evaluation detail", () => openDetail(evaluation, trigger));
    }
  }

  function setupDialog() {
    const dialog = $("#detail");
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
      state.validator = defaultValidator(Array.isArray(data.validators) ? data.validators : []);
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
