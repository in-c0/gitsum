// Definition hunt: finds where an identifier is defined, even in files the key-file index never read.
// Order: indexed chunks → files whose names look like the identifier → GitHub code search (token) → a
// bounded sweep of the remaining source files. raw.githubusercontent.com is not API-rate-limited,
// so the sweep costs time, not quota.

import { codeSearch, fetchRawFile } from "./github.js";
import { LANG_BY_EXT, TEST_PATH, isIgnored, isTextual } from "./analyze.js";

const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const extOf = (p) => (p.match(/\.([^./]+)$/)?.[1] || "").toLowerCase();

/** Splits isPlainObject / parse_args / HTTPServer / parse-args into lower-case parts. */
export function identParts(ident) {
  return String(ident)
    .split(/[_\-.$]|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])/)
    .filter(Boolean)
    .map((p) => p.toLowerCase());
}

/** The identifiers a question is about: `backticked`, CamelCase/snake_case words, or "where is X defined". */
export function questionIdentifiers(question) {
  const out = new Set();
  for (const m of question.matchAll(/`([^`\s]{2,80})`/g)) out.add(m[1].replace(/\(\)$/, "").split(/[.#:]/).pop());
  const where = question.match(/\b(?:where|which file)\b.*?\b(?:is|are|does)\s+(?:the\s+)?([A-Za-z_$][\w$]{1,60})(?:\(\))?\s+(?:defined|declared|implemented|live|located|come from|created|exported)/i);
  if (where) out.add(where[1]);
  const def = question.match(/\b(?:definition|implementation|source) of\s+(?:the\s+)?([A-Za-z_$][\w$]{1,60})/i);
  if (def) out.add(def[1]);
  for (const w of question.match(/[A-Za-z_$][\w$]{2,}/g) || []) {
    if (/[a-z][A-Z]|_[a-z]|^[A-Z][a-z]+[A-Z]/.test(w) || /\(\)$/.test(w)) out.add(w.replace(/\(\)$/, ""));
  }
  return [...out].filter((w) => !/^(where|what|which|defined|function|class|method)$/i.test(w));
}

const ID = (identifiers) => identifiers.map(escapeRe).join("|");

/** Definition shapes across common languages, strongest first. */
function definitionPatterns(identifiers) {
  const id = ID(identifiers);
  return [
    // function foo / def foo / class Foo / fn foo / func foo / struct / interface / type / enum / trait / module
    { w: 10, re: new RegExp(`(?:^|[\\s(])(?:export\\s+)?(?:default\\s+)?(?:declare\\s+)?(?:abstract\\s+)?(?:pub(?:\\([^)]*\\))?\\s+)?(?:async\\s+)?(?:static\\s+)?(?:function\\*?|def|class|interface|type|struct|enum|trait|fn|func|module|object|record|protocol|macro_rules!)\\s+(?:${id})\\b`) },
    // Go methods: func (r *T) Foo(
    { w: 10, re: new RegExp(`^func\\s+\\([^)]*\\)\\s+(?:${id})\\s*[([]`) },
    // exports.foo = / module.exports.foo = / Foo.prototype.foo =
    { w: 8, re: new RegExp(`(?:exports|module\\.exports|prototype)\\.(?:${id})\\s*=`) },
    // const/let/var foo =  ·  val foo =
    { w: 6, re: new RegExp(`(?:^|\\s)(?:export\\s+)?(?:const|let|var|val)\\s+(?:${id})\\s*(?::[^=]+)?=`) },
    // method shorthand: foo(a) {  ·  async foo() {  ·  foo: function
    { w: 6, re: new RegExp(`^\\s{0,8}(?:public\\s+|private\\s+|protected\\s+|static\\s+|async\\s+|override\\s+|get\\s+|set\\s+)*(?:${id})\\s*(?:\\([^)]*\\)\\s*(?::\\s*[^{]+)?\\{|:\\s*(?:async\\s+)?function)`) },
    // Java/C#/C/C++: ReturnType Foo(…) {
    { w: 4, re: new RegExp(`^\\s*(?:[\\w<>\\[\\],.*&:]+\\s+)+\\**(?:${id})\\s*\\([^;]*$`) },
  ];
}

/** Kept for callers that only need a yes/no test. */
export function definitionRegex(identifiers) {
  return new RegExp(definitionPatterns(identifiers).map((p) => p.re.source).join("|"), "m");
}

const CONTROL = /^\s*(?:if|for|while|switch|return|catch|else|elif|do|case|throw|raise|yield|assert|await|new|typeof|del|print|echo|go|defer|panic|except|with|not|lambda|super|import|from|using|include|require)\b/;

/** [{ line, score }] for every definition of the identifiers in this text (1-based lines). */
export function definitionMatches(text, identifiers) {
  if (!identifiers.length) return [];
  const pats = definitionPatterns(identifiers);
  const out = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.length > 400 || CONTROL.test(line)) continue;
    if (/^\s*(\/\/|#(?!\[)|\*|\/\*|--)/.test(line)) continue; // comments
    const p = pats.find((x) => x.re.test(line));
    if (!p) continue;
    const indent = line.match(/^\s*/)[0].replace(/\t/g, "    ").length;
    out.push({ line: i + 1, score: p.w + (indent === 0 ? 2 : indent <= 4 ? 0 : -2) + (/\bexport\b|\bpub\b/.test(line) ? 1 : 0) });
  }
  return out.sort((a, b) => b.score - a.score);
}

/** 1-based line numbers where any identifier is defined in this text. */
export function definitionLines(text, identifiers) {
  return definitionMatches(text, identifiers).map((m) => m.line);
}

function nameScore(path, parts) {
  const lower = path.toLowerCase();
  const base = lower.split("/").pop().replace(/\.[^.]+$/, "");
  const joined = parts.join("");
  const baseFlat = base.replace(/[-_.]/g, "");
  let s = 0;
  if (baseFlat === joined) s += 10;
  else if (baseFlat.includes(joined) || (joined.length > 3 && joined.includes(baseFlat) && baseFlat.length > 3)) s += 6;
  for (const p of parts) if (p.length > 2 && lower.includes(p)) s += 1.5;
  return s;
}

function isSource(f) {
  return LANG_BY_EXT[extOf(f.path)] && !isIgnored(f.path) && isTextual(f.path) && f.size > 0 && f.size < 300_000 && !/\.(css|scss|sass|less|html|md)$/i.test(f.path) && !/\.min\.js$/.test(f.path);
}

/** Adjusts a raw definition score by where the file lives. */
function placeScore(path, parts, testWanted) {
  let s = 0;
  if (/\.d\.(m|c)?ts$/.test(path)) s -= 6; // declarations, not implementations
  if (TEST_PATH.test(path) && !testWanted) s -= 5;
  if (/(^|\/)(examples?|docs?|fixtures?|testdata|bench(marks)?)\//.test(path)) s -= 3;
  const n = nameScore(path, parts);
  if (n >= 6) s += 3;
  else if (n >= 3) s += 1;
  return s;
}

async function scanFiles(repoData, paths, idents, opts, ctx, concurrency = 12) {
  const hits = [];
  let i = 0;
  async function worker() {
    while (i < paths.length && !(ctx.stopAt && hits.some((h) => h.score >= ctx.stopAt))) {
      const path = paths[i++];
      const text = await fetchRawFile(repoData, path, opts).catch(() => null);
      if (!text) continue;
      const m = definitionMatches(text, idents)[0];
      if (m) hits.push({ path, line: m.line, text, score: m.score + placeScore(path, ctx.parts, ctx.testWanted), via: ctx.via });
    }
  }
  await Promise.all(Array.from({ length: concurrency }, worker));
  return hits;
}

function snippetAt(text, line, before = 1, size = 12) {
  const lines = text.split("\n");
  const start = Math.max(0, line - 1 - before);
  return { startLine: start + 1, endLine: Math.min(lines.length, start + size), snippet: lines.slice(start, start + size).join("\n").trimEnd() };
}

const STRONG = 11; // a top-level function/class/type definition in a sensible place

/**
 * Finds where identifiers are defined. Returns [{ path, startLine, endLine, line, snippet, via }].
 * Gathers candidates from the index, then files named like the identifier, then code search (token),
 * then a bounded sweep — stopping as soon as a strong definition is found — and ranks them.
 * opts: { token, signal, budget } — budget caps the number of files fetched (default 160).
 */
export async function huntDefinition(repoData, index, identifiers, opts = {}) {
  const idents = identifiers.filter(Boolean);
  if (!idents.length) return [];
  const budget = opts.budget ?? 160;
  const parts = [...new Set(idents.flatMap(identParts))];
  const testWanted = idents.some((i) => /^test|Test$|_test$|Spec$/.test(i));
  const ctx = { parts, testWanted, stopAt: STRONG };
  const all = [];
  const strong = () => all.some((h) => h.score >= STRONG);

  // 1. Chunks already in the index (README + key files).
  const fileText = new Map();
  for (const d of index?.docs || []) {
    if (!fileText.has(d.path)) fileText.set(d.path, []);
    fileText.get(d.path).push(d);
  }
  for (const [path, chunks] of fileText) {
    for (const d of chunks) {
      const m = definitionMatches(d.text, idents)[0];
      if (!m) continue;
      const line = d.startLine + m.line - 1;
      const lines = d.text.split("\n");
      all.push({ path, line, score: m.score + placeScore(path, parts, testWanted), via: "index", startLine: Math.max(1, line - 1), endLine: line + 10, snippet: lines.slice(Math.max(0, m.line - 2), m.line + 10).join("\n").trimEnd() });
      break;
    }
  }

  const seen = new Set(fileText.keys());
  const sources = repoData.files.filter(isSource).filter((f) => !seen.has(f.path));
  let spent = 0;

  // 2. Files whose names look like the identifier.
  if (!strong()) {
    const byName = sources
      .map((f) => ({ f, s: nameScore(f.path, parts) }))
      .filter((x) => x.s >= 3)
      .sort((a, b) => b.s - a.s || a.f.path.length - b.f.path.length)
      .slice(0, 16)
      .map((x) => x.f.path);
    all.push(...(await scanFiles(repoData, byName, idents, opts, { ...ctx, via: "filename" })));
    byName.forEach((p) => seen.add(p));
    spent += byName.length;
  }

  // 3. GitHub code search (token only).
  if (!strong() && opts.token) {
    const found = (await codeSearch(repoData, idents[0], opts)).map((r) => r.path).filter((p) => !seen.has(p));
    all.push(...(await scanFiles(repoData, found, idents, opts, { ...ctx, via: "code-search" })));
    found.forEach((p) => seen.add(p));
    spent += found.length;
  }

  // 4. Bounded sweep: source dirs first, small-to-medium files, tests last (unless a test is wanted).
  if (!strong()) {
    const rest = sources
      .filter((f) => !seen.has(f.path))
      .map((f) => {
        let s = 0;
        if (/^(src|lib|pkg|internal|app|server|core|source|crates|packages|Sources)\//.test(f.path)) s += 3;
        if (TEST_PATH.test(f.path)) s += testWanted ? 1 : -6;
        if (/(^|\/)(examples?|docs?|scripts?|bench(marks)?|fixtures?|testdata)\//.test(f.path)) s -= 3;
        if (/\.d\.(m|c)?ts$/.test(f.path)) s -= 4;
        s -= f.path.split("/").length * 0.3;
        if (f.size > 120_000) s -= 1;
        return { f, s };
      })
      .sort((a, b) => b.s - a.s)
      .slice(0, Math.max(0, budget - spent))
      .map((x) => x.f.path);
    all.push(...(await scanFiles(repoData, rest, idents, opts, { ...ctx, via: "sweep" }, 16)));
  }

  all.sort((a, b) => b.score - a.score);
  const out = [];
  for (const h of all) {
    if (out.some((o) => o.path === h.path)) continue;
    out.push(h.text ? { path: h.path, line: h.line, ...snippetAt(h.text, h.line), via: h.via } : { path: h.path, line: h.line, startLine: h.startLine, endLine: h.endLine, snippet: h.snippet, via: h.via });
    if (out.length === 3) break;
  }
  return out;
}
