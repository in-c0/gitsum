// The one engine every Gitsum surface uses: open a repo once, then ask, find and read.

import { GitsumError, fetchRawFile, githubFileUrl, loadRepo, parseRepoInput } from "./github.js";
import { answerLocally, buildOverview, buildSearchIndex, suggestedQuestions } from "./analyze.js";
import { huntDefinition, questionIdentifiers } from "./find.js";

const sessions = new Map();

/**
 * Opens a repository: metadata, tree, overview and a search index over the key files.
 * opts: { token, signal, fetchImpl }
 */
export async function openRepo(input, opts = {}) {
  const parsed = typeof input === "string" ? parseRepoInput(input) : input;
  if (!parsed) throw new GitsumError("bad_input", "That doesn't look like a GitHub repository.", "Use owner/repo or a github.com URL.");
  const key = `${parsed.owner}/${parsed.repo}:${opts.token ? "t" : "a"}`.toLowerCase();
  const hit = sessions.get(key);
  if (hit && Date.now() - hit.t < 10 * 60 * 1000) return hit.v;

  const promise = (async () => {
    const repoData = opts.repoData || (await loadRepo(parsed.owner, parsed.repo, opts));
    const overview = await buildOverview(repoData, opts);
    const index = await buildSearchIndex(repoData, overview, opts);
    return { repoData, overview, index, opts };
  })();
  sessions.set(key, { t: Date.now(), v: promise });
  if (sessions.size > 50) sessions.delete(sessions.keys().next().value);
  promise.catch(() => sessions.delete(key));
  return promise;
}

const DEFINITION_Q = /\b(where|which file)\b.*\b(defined|declared|implemented|live|located|come from|exported)\b|\b(definition|implementation|source) of\b|`[^`]+`/i;

/**
 * Answers a question with evidence only (no AI): { text, code?, citations: [{ path, startLine?, endLine?, snippet?, note?, url }], kind }.
 * Definition questions trigger the definition hunt across the whole tree.
 */
export async function ask(session, question, opts = {}) {
  const { repoData, overview, index } = session;
  const merged = { ...session.opts, ...opts };
  const idents = questionIdentifiers(question);
  let result = null;

  if (idents.length && (DEFINITION_Q.test(question) || idents.some((w) => /[a-z][A-Z]|_/.test(w)))) {
    const defs = await huntDefinition(repoData, index, idents, merged);
    if (defs.length) {
      result = {
        text: `\`${idents[0]}\` is defined in ${defs[0].path}:${defs[0].line}.`,
        citations: defs.map((d) => ({ path: d.path, startLine: d.startLine, endLine: d.endLine, snippet: d.snippet, note: `definition (${d.via})` })),
        kind: "definition",
      };
    }
  }
  if (!result) result = answerLocally(question, repoData, overview, index);
  result.citations = (result.citations || []).map((c) => ({ ...c, url: githubFileUrl(repoData, c.path, c.startLine, c.endLine) }));
  return result;
}

/** Reads a file (optionally a line range) with line numbers. */
export async function readFile(session, path, { start = 1, end, maxLines = 220 } = {}) {
  const { repoData } = session;
  if (!repoData.files.some((f) => f.path === path)) {
    const close = closestPaths(repoData, path, 3);
    throw new GitsumError("no_file", `No file at ${path}.`, close.length ? `Did you mean: ${close.join(", ")}` : "List the tree to see what exists.");
  }
  const text = await fetchRawFile(repoData, path, session.opts);
  const lines = text.split("\n");
  const s = Math.max(1, start);
  const e = Math.min(lines.length, end ?? s + maxLines - 1, s + maxLines - 1);
  return { path, startLine: s, endLine: e, totalLines: lines.length, text: lines.slice(s - 1, e).join("\n"), url: githubFileUrl(repoData, path, s, e) };
}

/** Paths under a folder (or matching a glob-ish substring), capped. */
export function listTree(session, { under = "", match = "", limit = 200 } = {}) {
  const prefix = under.replace(/^\/+|\/+$/g, "");
  const needle = match.toLowerCase();
  const files = session.repoData.files.filter((f) => (!prefix || f.path.startsWith(`${prefix}/`)) && (!needle || f.path.toLowerCase().includes(needle)));
  return { total: files.length, files: files.slice(0, limit), truncated: files.length > limit || session.repoData.truncated };
}

function editDistance(a, b) {
  const dp = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let prev = dp[0];
    dp[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = dp[j];
      dp[j] = Math.min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return dp[b.length];
}

export function closestPaths(repoData, path, n = 3) {
  const target = path.toLowerCase();
  const base = target.split("/").pop();
  return repoData.files
    .map((f) => ({ p: f.path, d: editDistance(base, f.path.toLowerCase().split("/").pop()) + editDistance(target, f.path.toLowerCase()) * 0.1 }))
    .sort((a, b) => a.d - b.d)
    .slice(0, n)
    .map((x) => x.p);
}

export { suggestedQuestions, parseRepoInput, GitsumError };
