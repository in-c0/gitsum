// Gitsum side panel: Shape → Ask → Read, around the real tree drawn as strata.
// Runs inside the extension (chrome.* available) or as a plain page with ?fixture=… for the judge.
import { ask, closestPaths, openRepo, parseRepoInput, suggestedQuestions } from "../../src/core/engine.js";
import { GitsumError, loadRepo } from "../../src/core/github.js";
import { teamAsk, teamBody } from "../../src/core/team.js";
import { ANTHROPIC_MODEL, GRAIN_CAP, MAX_LAYERS, TEAM_API, TEAM_REQUEST_URL } from "./config.js";
import { LINGUIST, langColor, langOf } from "./colors.js";
import { Hero, prepare } from "./hero.js";
import { matchPaths } from "./match.js";
import * as store from "./store.js";
import { tick } from "./sound.js";

const $ = (id) => document.getElementById(id);
const IN_EXT = typeof chrome !== "undefined" && !!chrome.runtime?.id;
const params = new URLSearchParams(location.search);
const FIXTURE = params.get("fixture");

const state = {
  settings: { ghToken: "", anKey: "", teamKey: "", sound: true, theme: "system", lens: "strata", toured: {} },
  key: "", // owner/repo being shown
  tabId: null,
  tabUrl: "",
  repoData: null,
  session: null,
  sessionPromise: null,
  paths: [],
  lower: [],
  gIndex: new Map(), // file idx → grain index
  pathIdx: new Map(), // path → file idx
  match: { state: "empty", terms: [], lit: new Set() },
  closest: "",
  recent: [],
  history: [],
  cited: [],
  healthSel: -1,
  fetchImpl: undefined,
  fixtureLabel: "",
  timers: [],
  loadSeq: 0,
  inputAt: 0,
  litAt: 0,
};

// ---------------- boot ----------------

const hero = new Hero($("hero"), { onHover: showHover, onClick: grainClick });
hero.frameHooks.push(drawLineage);

init().catch((err) => fail(err));

async function init() {
  const saved = await store.get("settings", {});
  Object.assign(state.settings, saved);
  applyTheme();
  hero.readColors();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => hero.readColors());
  bindUI();
  setLensUI(state.settings.lens, false);
  hero.lens = state.settings.lens;

  if (FIXTURE) return loadFixture(FIXTURE);
  if (IN_EXT) {
    chrome.runtime.onMessage.addListener((msg) => {
      if (msg?.type === "gitsum:active") onActive(msg);
    });
    const active = await chrome.runtime.sendMessage({ type: "gitsum:active?" }).catch(() => null);
    return onActive(active || {});
  }
  const repo = params.get("repo");
  if (repo) return openLive(parseRepoInput(repo));
  showNonRepo();
}

function onActive({ tabId, url }) {
  state.tabId = tabId ?? state.tabId;
  state.tabUrl = url || "";
  const parsed = url && /^https:\/\/github\.com\//.test(url) ? parseRepoInput(url) : null;
  if (!parsed) return showNonRepo();
  const key = `${parsed.owner}/${parsed.repo}`.toLowerCase();
  if (key === state.key && state.repoData) return markHere();
  openLive(parsed);
}

// ---------------- loading ----------------

async function openLive(parsed) {
  if (!parsed) return showNonRepo();
  const seq = ++state.loadSeq;
  resetRepo(`${parsed.owner}/${parsed.repo}`);
  setHeader(parsed.owner, parsed.repo);
  const opts = { token: state.settings.ghToken || undefined, fetchImpl: state.fetchImpl };
  try {
    const repoData = await loadRepo(parsed.owner, parsed.repo, opts);
    if (seq !== state.loadSeq) return;
    showTree(repoData);
    state.sessionPromise = openRepo(parsed, { ...opts, repoData });
    const session = await state.sessionPromise;
    if (seq !== state.loadSeq) return;
    showOverview(session);
  } catch (err) {
    if (seq === state.loadSeq) fail(err);
  }
}

async function loadFixture(name) {
  // Simulated failures (labelled): every request answers like GitHub would.
  if (name === "404" || name === "ratelimit") {
    state.fetchImpl = async () =>
      name === "404"
        ? new Response('{"message":"Not Found"}', { status: 404 })
        : new Response('{"message":"API rate limit exceeded"}', { status: 403, headers: { "x-ratelimit-remaining": "0", "x-ratelimit-reset": String(Math.floor(Date.now() / 1000) + 17 * 60) } });
    return openLive({ owner: "fixture", repo: name === "404" ? "missing" : "limited" });
  }
  const file = name === "1" ? "repo" : name;
  const fx = await (await fetch(`fixtures/${encodeURIComponent(file)}.json`)).json();
  const repoData = { ...fx.repoData };
  if (params.get("partial") === "1") repoData.truncated = true;
  state.fixtureLabel = fx.kind === "clone-snapshot" ? "clone snapshot" : "snapshot";
  const raw = fx.raw || {};
  state.fetchImpl = async (url) => {
    const m = String(url).match(/^https:\/\/raw\.githubusercontent\.com\/[^/]+\/[^/]+\/[^/]+\/(.+)$/);
    if (!m) return new Response('{"message":"offline fixture"}', { status: 404 });
    const path = m[1].split("/").map(decodeURIComponent).join("/");
    if (raw[path] !== undefined) return new Response(raw[path]);
    if (fx.rawBase) return fetch(fx.rawBase + path.split("/").map(encodeURIComponent).join("/"));
    return new Response("404", { status: 404 });
  };
  window.__gitsum = { hero, state, fixture: fx };
  const seq = ++state.loadSeq;
  resetRepo(repoData.fullName);
  setHeader(repoData.owner, repoData.repo);
  showTree(repoData);
  state.sessionPromise = openRepo({ owner: repoData.owner, repo: repoData.repo }, { repoData, fetchImpl: state.fetchImpl });
  try {
    const session = await state.sessionPromise;
    if (seq === state.loadSeq) showOverview(session);
  } catch (err) {
    fail(err);
  }
}

function resetRepo(key) {
  state.timers.forEach(clearTimeout);
  state.timers = [];
  state.key = key.toLowerCase();
  state.repoData = null;
  state.session = null;
  state.sessionPromise = null;
  state.history = [];
  state.cited = [];
  state.match = { state: "empty", terms: [], lit: new Set() };
  $("q").value = "";
  closeAnswer();
  hideNotice();
  setAct(0);
  document.body.dataset.state = "loading";
  $("badges").textContent = "";
  $("nFiles").textContent = "0";
  $("nFolders").textContent = "0";
  delete $("counter").dataset.final;
  hero.setIdle();
}

function setHeader(owner, repo) {
  $("repoOwner").textContent = owner ? `${owner}/` : "";
  $("repoName").textContent = repo;
  $("repo").title = `${owner}/${repo}`;
}

function later(fn, ms) {
  state.timers.push(setTimeout(fn, ms));
}

// ---------------- act 1: shape ----------------

function showTree(repoData) {
  state.repoData = repoData;
  state.paths = repoData.files.map((f) => f.path);
  state.lower = state.paths.map((p) => p.toLowerCase());
  state.pathIdx = new Map(state.paths.map((p, i) => [p, i]));
  const data = prepare(repoData.files, { cap: GRAIN_CAP, maxLayers: MAX_LAYERS });
  state.gIndex = new Map(data.grains.map((g, gi) => [g.idx, gi]));
  document.body.dataset.state = "ready";
  hero.setData(data);

  const badges = [];
  if (repoData.truncated) badges.push(["partial", "GitHub truncated this tree: not every file is listed", true]);
  if (data.capped) badges.push([`top ${data.grains.length.toLocaleString()} by size`, `${data.total.toLocaleString()} files; the largest ${data.grains.length.toLocaleString()} are drawn`, true]);
  if (repoData.private) badges.push(["private", "Private repository"]);
  if (repoData.archived) badges.push(["archived", "Archived repository"]);
  if (state.fixtureLabel) badges.push([state.fixtureLabel, "Fixture data recorded from a real repository"]);
  $("badges").replaceChildren(
    ...badges.map(([t, title, warn]) => {
      const b = el("span", `badge${warn ? " warn" : ""}`, t);
      b.title = title;
      return b;
    })
  );

  countUp($("nFiles"), repoData.files.length);
  countUp($("nFolders"), folderCount(state.paths), () => ($("counter").dataset.final = "1"));
  setAct(1);
  markHere();
  loadRecent();
  later(() => setAct(2), 950);
}

function folderCount(paths) {
  const dirs = new Set();
  for (const p of paths) {
    let i = p.lastIndexOf("/");
    while (i > 0) {
      const d = p.slice(0, i);
      if (dirs.has(d)) break;
      dirs.add(d);
      i = d.lastIndexOf("/");
    }
  }
  return dirs.size;
}

function countUp(node, target, done) {
  const t0 = performance.now();
  const dur = 900;
  const step = () => {
    const t = Math.min(1, (performance.now() - t0) / dur);
    const v = t >= 1 ? target : Math.floor(target * (1 - Math.pow(1 - t, 3)));
    node.textContent = v.toLocaleString("en-US");
    if (t < 1) requestAnimationFrame(step);
    else done?.();
  };
  requestAnimationFrame(step);
}

function markHere() {
  // "you are here": the file open in the GitHub tab, if any
  const m = state.tabUrl.match(/^https:\/\/github\.com\/[^/]+\/[^/]+\/blob\/(.+?)(?:[?#].*)?$/);
  let here = -1;
  if (m && state.repoData) {
    const segs = m[1].split("/").map(decodeURIComponent);
    for (let k = 1; k < segs.length && here < 0; k++) {
      const idx = state.pathIdx.get(segs.slice(k).join("/"));
      if (idx !== undefined) here = state.gIndex.get(idx) ?? -1;
    }
  }
  hero.setHere(here);
}

// ---------------- act 3: read ----------------

function showOverview(session) {
  state.session = session;
  const { overview: o, repoData: r } = session;

  // languages bar (bytes, counted from the tree by the engine)
  const langs = o.languages || [];
  $("langBar").replaceChildren(
    ...langs.map((l) => {
      const i = el("i");
      i.style.width = `${l.pct}%`;
      i.style.background = LINGUIST[l.name] || "var(--grain)";
      i.title = `${l.name} ${l.pct.toFixed(1)}%`;
      return i;
    })
  );
  $("langLegend").replaceChildren(
    ...(langs.length
      ? langs.slice(0, 3).map((l) => {
          const s = el("span");
          const dot = el("i");
          dot.style.background = LINGUIST[l.name] || "var(--grain)";
          s.append(dot, `${l.name} ${l.pct < 10 ? l.pct.toFixed(1) : Math.round(l.pct)}%`);
          return s;
        })
      : [el("span", "", "no source languages detected")])
  );

  // health dots
  $("dots").replaceChildren(
    ...o.health.map((h, i) => {
      const b = el("button", `hd ${h.ok ? "ok" : "no"}`);
      b.setAttribute("aria-label", `${h.label}: ${h.value}`);
      b.title = `${h.label}: ${h.value}`;
      b.addEventListener("click", () => selectHealth(state.healthSel === i ? -1 : i));
      return b;
    })
  );
  selectHealth(-1);

  // stack chips (+N opens a fitted popover)
  renderStack(o.stack);

  // run
  const run = $("run");
  if (o.run?.code) {
    run.replaceChildren(
      ...o.run.code.split("\n").flatMap((line, i) => {
        const out = [];
        if (i) out.push("\n");
        out.push(el("span", "p", "$ "), line);
        return out;
      })
    );
    run.title = o.run.source;
    $("copyRun").hidden = false;
  } else {
    run.replaceChildren(el("span", "p", "no run command found in README or package.json"));
    $("copyRun").hidden = true;
  }

  // live stamp
  const live = $("roLive");
  const parts = [];
  if (r.stars !== null && r.stars !== undefined) parts.push(`★ ${Number(r.stars).toLocaleString("en-US")}`);
  if (r.pushedAt) parts.push(`pushed ${ago(r.pushedAt)}`);
  if (r.license) parts.push(r.license);
  const stamp = el("span", state.fixtureLabel ? "" : "live", state.fixtureLabel ? `${state.fixtureLabel} ${hhmm(r.fetchedAt)}` : `live ${hhmm(r.fetchedAt)}`);
  stamp.title = `Fetched from GitHub at ${new Date(r.fetchedAt).toLocaleString()}`;
  live.replaceChildren(stamp, parts.length ? ` · ${parts.join(" · ")}` : "");

  if ($("q").value === "") renderSelection();
  const reveal = () => {
    setAct(3);
    later(firstRunTour, 900);
  };
  if (Number(document.body.dataset.act) >= 2) later(reveal, 350);
  else later(reveal, 1350);
}

function renderStack(stack) {
  const box = $("stack");
  box.replaceChildren();
  if (!stack.length) {
    box.append(el("span", "sc more", "no manifest found"));
    return;
  }
  const chips = stack.map((s) => el("span", "sc", s));
  box.append(...chips);
  // fit: drop chips that overflow, then add "+N"
  requestAnimationFrame(() => {
    const max = box.clientWidth;
    let hidden = 0;
    while (box.scrollWidth > max && box.children.length > 1) {
      box.lastChild.remove();
      hidden++;
      if (hidden === 1) continue;
    }
    if (hidden) {
      const more = el("button", "sc more", `+${hidden}`);
      box.append(more);
      while (box.scrollWidth > max && box.children.length > 2) {
        box.children[box.children.length - 2].remove();
        hidden++;
        more.textContent = `+${hidden}`;
      }
      more.addEventListener("click", (e) => {
        e.stopPropagation();
        openPopover(more, stack.map((s) => el("span", "sc", s)));
      });
    }
  });
}

function selectHealth(i) {
  state.healthSel = i;
  const o = state.session?.overview;
  if (!o) return;
  [...$("dots").children].forEach((d, k) => d.classList.toggle("sel", k === i));
  const cap = $("healthCaption");
  if (i < 0) {
    const ok = o.health.filter((h) => h.ok).length;
    cap.replaceChildren(el("b", "", `${ok}/${o.health.length}`), " healthy");
  } else {
    const h = o.health[i];
    cap.replaceChildren(el("b", "", h.label), ` ${h.value}`);
  }
}

function firstRunTour() {
  const t = state.settings.toured || {};
  if (!t.lens) {
    // flip the lens once so its meaning is seen, then return; silent, stored so it happens once
    const back = state.settings.lens;
    const other = back === "strata" ? "atlas" : "strata";
    later(() => switchLens(other, { silent: true, persist: false }), 200);
    later(() => switchLens(back, { silent: true, persist: false }), 1500);
    t.lens = true;
  }
  if (!t.health && state.session) {
    const firstBad = state.session.overview.health.findIndex((h) => !h.ok);
    later(() => selectHealth(firstBad >= 0 ? firstBad : 0), 2300);
    later(() => selectHealth(-1), 3600);
    t.health = true;
  }
  state.settings.toured = t;
  saveSettings();
  later(() => (document.body.dataset.toured = "1"), 3700);
}

// ---------------- act 2: ask ----------------

let debounce = 0;
function onInput() {
  state.inputAt = performance.now();
  clearTimeout(debounce);
  debounce = setTimeout(runMatch, 45);
}

function runMatch() {
  const q = $("q").value;
  const m = matchPaths(state.paths, state.lower, q);
  state.match = m;
  state.closest = "";
  if (m.state === "none" && state.repoData && q.trim().length >= 3) {
    const word = (m.terms.length ? m.terms.reduce((a, b) => (b.length > a.length ? b : a)) : q.trim()).slice(0, 60);
    state.closest = closestPaths(state.repoData, word, 1)[0] || "";
  }
  hero.setLit(m.state === "ok" ? m.lit : m.state === "none" ? new Set() : null);
  renderSelection();
  state.litAt = performance.now();
  const sel = $("selection");
  sel.dataset.lit = String(m.state === "ok" ? m.lit.size : 0);
  sel.dataset.q = q;
  sel.dataset.ms = String(Math.round(state.litAt - state.inputAt));
}

function renderSelection() {
  const sel = $("selection");
  const m = state.match;
  const q = $("q").value;
  sel.dataset.state = m.state;
  if (m.state === "empty" || !q.trim()) {
    const kids = [];
    if (state.recent.length) {
      kids.push(el("span", "label", "recent"));
      state.recent.forEach((r, i) => {
        const chip = el("span", "recent");
        const t = el("span", "", r);
        t.title = r;
        t.addEventListener("click", () => submit(r));
        const x = el("button", "x", "×");
        x.setAttribute("aria-label", `Forget "${r}"`);
        x.addEventListener("click", (e) => {
          e.stopPropagation();
          state.recent.splice(i, 1);
          saveRecent();
          renderSelection();
        });
        chip.append(t, x);
        kids.push(chip);
      });
    } else if (state.session) {
      for (const s of suggestedQuestions(state.session.overview).slice(0, 3)) {
        const b = el("button", "recent suggest", s);
        b.addEventListener("click", () => submit(s));
        kids.push(b);
      }
    }
    sel.replaceChildren(...kids);
    return;
  }
  if (m.state === "short") {
    sel.replaceChildren(el("span", "", "keep typing"));
    return;
  }
  if (m.state === "none") {
    const kids = [el("span", "", "no path match")];
    if (state.closest) kids.push(el("span", "label", "· closest"), el("b", "", state.closest), el("span", "tab", "Tab"));
    sel.replaceChildren(...kids);
    return;
  }
  const drawn = [...m.lit].filter((i) => state.gIndex.has(i)).length;
  const kids = [el("span", "", "lit:"), el("b", "", m.lit.size.toLocaleString("en-US")), el("span", "", m.lit.size === 1 ? "file" : "files")];
  if (drawn !== m.lit.size) kids.push(el("span", "label", `(${drawn.toLocaleString("en-US")} drawn)`));
  for (const t of m.terms) {
    kids.push(el("span", "label", "·"), el("span", "tok", `‘${t}’`));
    const x = el("button", "x", "×");
    x.setAttribute("aria-label", `Remove ${t}`);
    x.addEventListener("click", () => removeTerm(t));
    kids.push(x);
  }
  sel.replaceChildren(...kids);
}

function removeTerm(term) {
  const input = $("q");
  const re = new RegExp(`(^|[^a-z0-9_$.\\-])${term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?=$|[^a-z0-9_$.\\-])`, "i");
  input.value = input.value.replace(re, "$1").replace(/\s{2,}/g, " ").trim();
  runMatch();
  input.focus();
}

async function submit(question) {
  const q = question.trim();
  if (!q || !state.repoData) return;
  $("q").value = q;
  runMatch();
  rememberQuestion(q);
  openAnswer("thinking…", "reading the evidence");
  try {
    const session = state.session || (await state.sessionPromise);
    if (!session) return;
    const result = await ask(session, q, { fetchImpl: state.fetchImpl, token: state.settings.ghToken || undefined });
    if ($("q").value.trim() !== q && $("answer").dataset.q !== q) return;
    showAnswer(q, result, session);
  } catch (err) {
    openAnswer(err.message || "Something went wrong.", "error");
    if (err.hint) $("answerText").append(`\n${err.hint}`);
  }
}

function openAnswer(text, kind) {
  const box = $("answer");
  box.hidden = false;
  box.dataset.q = $("q").value.trim();
  $("answerKind").textContent = kind;
  $("answerText").textContent = text;
  $("answerCode").hidden = true;
  $("chips").replaceChildren();
  state.cited = [];
  hero.setCited(new Set());
  drawLineage();
}

function closeAnswer() {
  $("answer").hidden = true;
  state.cited = [];
  hero.setCited(new Set());
  drawLineage();
}

function showAnswer(q, result, session) {
  $("answerKind").textContent = `evidence · ${result.kind}`;
  setRichText($("answerText"), result.text);
  if (result.code) {
    $("answerCode").hidden = false;
    $("answerCode").textContent = result.code;
  }
  const cites = (result.citations || []).slice(0, 6);
  state.cited = cites.map((c) => ({ ...c, gi: state.gIndex.get(state.pathIdx.get(c.path)) ?? -1 }));
  $("chips").replaceChildren(
    ...state.cited.map((c) => {
      const b = el("button", "chip");
      b.title = c.note ? `${c.path} — ${c.note}` : c.path;
      b.append(el("i"), el("span", "", c.startLine ? `${c.path}:${c.startLine}` : c.path));
      b.addEventListener("click", () => openUrl(c.url));
      return b;
    })
  );
  hero.setCited(new Set(state.cited.map((c) => c.gi).filter((gi) => gi >= 0)));
  requestAnimationFrame(drawLineage);
  state.history.push({ role: "user", text: q }, { role: "assistant", text: result.text });
  state.history = state.history.slice(-8);
  aiAnswer(q, result, session);
}

async function aiAnswer(q, result, session) {
  const { teamKey, anKey } = state.settings;
  const useTeam = TEAM_API && teamKey;
  if (!useTeam && !anKey) return;
  const text = $("answerText");
  let got = "";
  const onText = (t) => {
    if (!got) text.textContent = "";
    got += t;
    setRichText(text, got);
  };
  try {
    if (useTeam) {
      $("answerKind").replaceChildren(el("span", "live", "team"), " · streaming");
      const res = await teamAsk({ base: TEAM_API.replace(/\/$/, ""), key: teamKey }, teamBody(session, q, result.citations, state.history.slice(0, -2)), { onText });
      $("answerKind").replaceChildren(el("span", "live", "team"), res.limit ? ` · ${res.used}/${res.limit} this month` : "");
    } else {
      $("answerKind").replaceChildren(el("span", "live", ANTHROPIC_MODEL), " · your key · streaming");
      await anthropicStream(anKey, q, result, session, onText);
      $("answerKind").replaceChildren(el("span", "live", ANTHROPIC_MODEL), " · your key");
    }
  } catch (err) {
    $("answerKind").textContent = `evidence · AI failed: ${err.message || err.code || "error"}`.slice(0, 120);
    if (!got) setRichText(text, result.text);
  }
}

function evidence(session, result) {
  const { repoData: r, overview: o } = session;
  const lines = [`Repository: ${r.fullName} (branch ${r.branch})`];
  if (o.oneLiner) lines.push(`Description: ${o.oneLiner}`);
  if (o.stack.length) lines.push(`Stack: ${o.stack.join(", ")}`);
  if (o.languages.length) lines.push(`Languages: ${o.languages.map((l) => `${l.name} ${l.pct.toFixed(0)}%`).join(", ")}`);
  lines.push(`Top-level folders: ${o.structure.dirs.map((d) => `${d.name}/ (${d.count})`).join(", ")}`);
  lines.push(`Key files: ${o.keyFiles.map((f) => `${f.path} (${f.reason})`).join(", ")}`);
  const numbered = (path, start, text) => `--- ${path} ---\n${text.split("\n").map((l, i) => `${start + i}: ${l}`).join("\n")}`;
  if (o.readme && o.readmePath) lines.push(numbered(o.readmePath, 1, o.readme.split("\n").slice(0, 120).join("\n").slice(0, 6000)));
  for (const c of result.citations || []) if (c.snippet) lines.push(numbered(c.path, c.startLine || 1, c.snippet));
  lines.push(`Evidence answer: ${result.text}`);
  return lines.join("\n\n");
}

async function anthropicStream(key, q, result, session, onText) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
      "anthropic-dangerous-direct-browser-access": "true",
    },
    body: JSON.stringify({
      model: ANTHROPIC_MODEL,
      max_tokens: 700,
      stream: true,
      system:
        "You explain a GitHub repository using ONLY the evidence given. Every evidence line starts with its line number. " +
        "Cite as path:line or path:start-end, and only paths and lines that appear in the evidence; never invent files, lines or numbers. " +
        "If the evidence does not answer the question, say so in one sentence and name the closest cited file. " +
        "Be brief: at most 6 short sentences or a short list. Plain text; wrap identifiers in backticks.",
      messages: [
        ...state.history.slice(0, -2).map((h) => ({ role: h.role, content: h.text })),
        { role: "user", content: `${evidence(session, result)}\n\nQuestion: ${q}` },
      ],
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.error?.message || `Anthropic API ${res.status}`);
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      const data = frame.match(/^data: (.*)$/m)?.[1];
      if (!data) continue;
      const obj = JSON.parse(data);
      if (obj.type === "content_block_delta" && obj.delta?.type === "text_delta") onText(obj.delta.text);
      if (obj.type === "error") throw new Error(obj.error?.message || "stream error");
    }
  }
}

// ---------------- recent questions ----------------

async function loadRecent() {
  const key = state.key;
  const list = await store.get(`recent:${key}`, []);
  if (key !== state.key) return;
  state.recent = Array.isArray(list) ? list : [];
  if (!$("q").value) renderSelection();
}
function rememberQuestion(q) {
  state.recent = [q, ...state.recent.filter((r) => r.toLowerCase() !== q.toLowerCase())].slice(0, 6);
  saveRecent();
}
function saveRecent() {
  store.set(`recent:${state.key}`, state.recent);
}

// ---------------- lineage threads ----------------

function drawLineage() {
  const svg = $("lineage");
  if ($("answer").hidden || !state.cited.length) {
    if (svg.firstChild) svg.replaceChildren();
    return;
  }
  const cr = $("hero").getBoundingClientRect();
  const chips = [...$("chips").children];
  const strip = $("chips").getBoundingClientRect();
  const NS = "http://www.w3.org/2000/svg";
  const out = [];
  state.cited.forEach((c, i) => {
    const chip = chips[i];
    if (!chip || c.gi < 0) return;
    const p = hero.center(c.gi);
    if (!p) return;
    const r = chip.getBoundingClientRect();
    if (r.right < strip.left || r.left > strip.right) return;
    const box = $("answer").getBoundingClientRect();
    const x2 = cr.left + p.x;
    const y2 = cr.top + p.y;
    const path = document.createElementNS(NS, "path");
    if (box.left > cr.right - 4) {
      // side-by-side layout: leave the answer block's left edge at the chip's height
      const x1 = box.left;
      const y1 = r.top + r.height / 2;
      const mx = (x1 + x2) / 2;
      path.setAttribute("d", `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`);
    } else {
      // stacked layout: emerge from the answer block's top edge above the chip
      const x1 = r.left + 11;
      const y1 = box.top;
      const my = (y1 + y2) / 2;
      path.setAttribute("d", `M${x1},${y1} C${x1},${my} ${x2},${my} ${x2},${y2}`);
    }
    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", x2);
    dot.setAttribute("cy", y2);
    dot.setAttribute("r", 1.8);
    out.push(path, dot);
  });
  svg.replaceChildren(...out);
}

// ---------------- hero interaction ----------------

function showHover(g) {
  const h = $("hover");
  if (!g) {
    h.classList.remove("on");
    return;
  }
  const dot = el("i");
  dot.style.background = langColor(g.lang, "var(--grain)");
  h.replaceChildren(dot, el("b", "", g.path), ` · ${bytes(g.size)}${g.lang ? ` · ${g.lang}` : ""}`);
  h.classList.add("on");
}

function grainClick(g) {
  // a click advances the default focus: this file's layer ↔ all layers
  hero.setFocus(hero.focusLayer === g.layer ? -1 : g.layer);
}

// ---------------- lens ----------------

function switchLens(lens, { silent = false, persist = true } = {}) {
  const current = $("lens").querySelector('[aria-selected="true"]')?.dataset.lens;
  if (hero.lens === lens && current === lens) return;
  hero.setLens(lens);
  setLensUI(lens, true);
  if (!silent) tick(!state.settings.sound);
  if (persist) {
    state.settings.lens = lens;
    saveSettings();
  }
}

function setLensUI(lens) {
  for (const b of $("lens").querySelectorAll("button")) b.setAttribute("aria-selected", String(b.dataset.lens === lens));
  placeHairline();
}
function placeHairline() {
  const sel = $("lens").querySelector('[aria-selected="true"]');
  const hl = $("lens").querySelector(".hairline");
  if (!sel) return;
  hl.style.width = `${sel.offsetWidth}px`;
  hl.style.transform = `translateX(${sel.offsetLeft - 2}px)`;
}

// ---------------- notices / failures ----------------

function showNonRepo() {
  state.loadSeq++;
  resetRepo("");
  setHeader("", "Gitsum");
  document.body.dataset.state = "idle";
  notice("Open a GitHub repository", "The strata fill in from its file tree.");
}

function fail(err) {
  const e = err instanceof GitsumError || err?.code ? err : new GitsumError("error", err?.message || "Something went wrong.", "Try again.");
  document.body.dataset.state = "error";
  document.body.dataset.error = e.code;
  hero.setIdle();
  setAct(0);
  const tokenFix = ["not_found", "rate_limited", "bad_token", "blocked"].includes(e.code);
  notice(e.message, e.hint || "", tokenFix ? [state.settings.ghToken ? "Edit GitHub token" : "Add GitHub token", () => openDrawer("ghToken")] : ["Retry", () => location.reload()]);
}

function notice(title, hint, action) {
  $("notice").hidden = false;
  $("noticeTitle").textContent = title;
  $("noticeHint").textContent = hint;
  const btn = $("noticeAction");
  btn.hidden = !action;
  if (action) {
    btn.textContent = action[0];
    btn.onclick = action[1];
  }
}
function hideNotice() {
  $("notice").hidden = true;
  delete document.body.dataset.error;
}

// ---------------- settings ----------------

function openDrawer(focusId) {
  $("drawer").hidden = false;
  $("ghToken").value = state.settings.ghToken || "";
  $("anKey").value = state.settings.anKey || "";
  $("teamKey").value = state.settings.teamKey || "";
  const team = $("teamKey");
  if (TEAM_API) {
    team.disabled = false;
    $("teamNote").textContent = "Hosted AI answers for your team, no personal key.";
  } else {
    team.disabled = true;
    const a = el("a", "", "early access");
    a.href = TEAM_REQUEST_URL;
    a.target = "_blank";
    a.rel = "noopener";
    $("teamNote").replaceChildren("Team plan — ", a);
  }
  $("sound").setAttribute("aria-checked", String(!!state.settings.sound));
  for (const b of $("themes").children) b.setAttribute("aria-checked", String(b.dataset.themeSet === state.settings.theme));
  if (focusId) $(focusId).focus();
}

function closeDrawer() {
  const before = state.settings.ghToken;
  state.settings.ghToken = $("ghToken").value.trim();
  state.settings.anKey = $("anKey").value.trim();
  if (TEAM_API) state.settings.teamKey = $("teamKey").value.trim();
  saveSettings();
  $("drawer").hidden = true;
  if (before !== state.settings.ghToken && !FIXTURE) {
    const parsed = state.key ? parseRepoInput(state.key) : null;
    if (parsed) openLive(parsed);
  }
}

function saveSettings() {
  store.set("settings", state.settings);
}

function applyTheme() {
  document.documentElement.dataset.theme = state.settings.theme || "system";
}

// ---------------- UI bindings ----------------

function bindUI() {
  const q = $("q");
  q.addEventListener("input", onInput);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      clearTimeout(debounce);
      submit(q.value);
    } else if (e.key === "Tab" && state.match.state === "none" && state.closest) {
      e.preventDefault();
      q.value = state.closest.split("/").pop();
      runMatch();
    } else if (e.key === "Escape") {
      if (q.value) {
        q.value = "";
        runMatch();
      } else closeAnswer();
    }
  });
  for (const b of $("lens").querySelectorAll("button")) b.addEventListener("click", () => switchLens(b.dataset.lens));
  $("gear").addEventListener("click", () => ($("drawer").hidden ? openDrawer() : closeDrawer()));
  $("drawerClose").addEventListener("click", closeDrawer);
  $("answerClose").addEventListener("click", closeAnswer);
  $("chips").addEventListener("scroll", drawLineage, { passive: true });
  $("answerBody").addEventListener("scroll", drawLineage, { passive: true });
  $("sound").addEventListener("click", () => {
    state.settings.sound = !state.settings.sound;
    $("sound").setAttribute("aria-checked", String(state.settings.sound));
    tick(!state.settings.sound);
    saveSettings();
  });
  for (const b of $("themes").children)
    b.addEventListener("click", () => {
      state.settings.theme = b.dataset.themeSet;
      for (const o of $("themes").children) o.setAttribute("aria-checked", String(o === b));
      applyTheme();
      hero.readColors();
      saveSettings();
    });
  $("copyRun").addEventListener("click", async () => {
    const code = state.session?.overview.run?.code;
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      const ta = el("textarea");
      ta.value = code;
      document.body.append(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    $("copyRun").classList.add("done");
    setTimeout(() => $("copyRun").classList.remove("done"), 1200);
  });
  document.addEventListener("click", (e) => {
    if (!$("popover").hidden && !$("popover").contains(e.target)) $("popover").hidden = true;
  });
  new ResizeObserver(() => {
    placeHairline();
    drawLineage();
    if (state.session) renderStack(state.session.overview.stack);
  }).observe(document.body);
}

function openPopover(anchor, children) {
  const pop = $("popover");
  pop.replaceChildren(...children);
  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const pr = pop.getBoundingClientRect();
  const left = Math.max(8, Math.min(innerWidth - pr.width - 8, r.left));
  const top = r.top - pr.height - 6 > 8 ? r.top - pr.height - 6 : Math.min(innerHeight - pr.height - 8, r.bottom + 6);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
}

function openUrl(url) {
  if (!url) return;
  if (IN_EXT && state.tabId != null) chrome.tabs.update(state.tabId, { url }).catch(() => chrome.tabs.create({ url }));
  else if (FIXTURE) window.__gitsumOpened = url;
  else window.open(url, "_blank", "noopener");
}

function setAct(n) {
  document.body.dataset.act = String(n);
}

// ---------------- helpers ----------------

function el(tag, cls = "", text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}

function setRichText(node, text) {
  // `code` spans only; everything else stays plain text
  const parts = String(text).split(/(`[^`\n]+`)/g);
  node.replaceChildren(...parts.map((p) => (p.startsWith("`") && p.endsWith("`") && p.length > 2 ? el("code", "", p.slice(1, -1)) : document.createTextNode(p))));
}

function bytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(n < 10240 ? 1 : 0)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
function hhmm(iso) {
  const d = iso ? new Date(iso) : new Date();
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
function ago(iso) {
  const days = Math.floor((Date.now() - new Date(iso).getTime()) / 86400000);
  if (days < 1) return "today";
  if (days < 60) return `${days}d ago`;
  if (days < 730) return `${Math.round(days / 30)}mo ago`;
  return `${Math.round(days / 365)}y ago`;
}

// for the language of a path in other modules / debugging
export { langOf };
