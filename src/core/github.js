// GitHub access shared by every Gitsum surface (extension, MCP, CLI, Skill).
// Runs in browsers, extensions and Node 18+ (global fetch). No dependencies.
// - repo metadata + tree: REST API (60 req/h anonymous, 5,000/h with a token; 2 calls per repo)
// - file contents: raw.githubusercontent.com for public repos (not rate-limited like the API),
//   the contents API for private repos (needs a token)

const API = "https://api.github.com";
const RAW = "https://raw.githubusercontent.com";
const CACHE_TTL_MS = 10 * 60 * 1000;
const memory = new Map();

export class GitsumError extends Error {
  constructor(code, message, hint) {
    super(message);
    this.name = "GitsumError";
    this.code = code;
    this.hint = hint;
  }
}

const OWNER_RE = /^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$/;
const REPO_RE = /^[A-Za-z0-9._-]{1,100}$/;

/**
 * Accepts every common way people paste a repo:
 * https://github.com/o/r, github.com/o/r/tree/main/src, o/r, git@github.com:o/r.git, gitsum URLs.
 * Returns { owner, repo } or null.
 */
export function parseRepoInput(input) {
  if (!input) return null;
  let s = String(input).trim();
  if (!s) return null;
  s = s.replace(/^git\+/, "");
  const ssh = s.match(/^git@github\.com:([^/\s]+)\/([^/\s]+?)(?:\.git)?\/?$/i);
  let owner;
  let repo;
  if (ssh) {
    [, owner, repo] = ssh;
  } else {
    s = s.replace(/^https?:\/\//i, "").replace(/^www\./i, "");
    if (/^github\.com\//i.test(s)) s = s.slice("github.com/".length);
    else if (s.split("/")[0].includes(".")) s = s.split("/").slice(1).join("/"); // gitsum or other mirror host: keep /o/r
    s = s.split(/[?#]/)[0];
    const parts = s.split("/").filter(Boolean);
    if (parts.length < 2) return null;
    [owner, repo] = parts;
    repo = repo.replace(/\.git$/i, "");
  }
  if (!OWNER_RE.test(owner) || !REPO_RE.test(repo) || repo === "." || repo === "..") return null;
  if (RESERVED_OWNERS.has(owner.toLowerCase())) return null;
  return { owner, repo };
}

// github.com/<these>/… are product pages, not repositories.
const RESERVED_OWNERS = new Set(
  "settings orgs marketplace explore topics trending collections sponsors notifications pulls issues codespaces features about pricing login signup new organizations enterprise apps search users site security readme customer-stories team".split(" ")
);

function cacheGet(key) {
  const hit = memory.get(key);
  if (hit && Date.now() - hit.t < CACHE_TTL_MS) return hit.v;
  return null;
}
function cacheSet(key, v) {
  memory.set(key, { t: Date.now(), v });
  if (memory.size > 200) memory.delete(memory.keys().next().value);
}

/** Test hook. */
export function __clearCache() {
  memory.clear();
}

function headersFor(token, accept = "application/vnd.github+json") {
  const h = { Accept: accept, "X-GitHub-Api-Version": "2022-11-28" };
  if (token) h.Authorization = `Bearer ${token}`;
  return h;
}

function minutesUntil(epochSeconds) {
  if (!epochSeconds) return null;
  const mins = Math.max(1, Math.ceil((epochSeconds * 1000 - Date.now()) / 60000));
  return mins <= 60 ? mins : null;
}

async function apiGet(path, { token, signal, fetchImpl = globalThis.fetch } = {}) {
  let res;
  try {
    res = await fetchImpl(`${API}${path}`, { signal, headers: headersFor(token) });
  } catch (err) {
    if (err?.name === "AbortError") throw err;
    throw new GitsumError("network", "Couldn't reach GitHub.", "Check your connection and try again.");
  }
  if (res.ok) return res.json();
  const remaining = res.headers.get("x-ratelimit-remaining");
  if ((res.status === 403 || res.status === 429) && (remaining === "0" || res.status === 429)) {
    const mins = minutesUntil(Number(res.headers.get("x-ratelimit-reset")));
    throw new GitsumError(
      "rate_limited",
      "GitHub's hourly API limit was reached.",
      `${mins ? `It resets in about ${mins} min` : "It resets within the hour"}.${token ? "" : " Add a GitHub token to raise the limit from 60 to 5,000 requests an hour."}`
    );
  }
  if (res.status === 404) {
    throw new GitsumError(
      "not_found",
      "Repository not found.",
      token ? "Check the spelling, and that your token can read this repository." : "It may be private, renamed or misspelled. Private repos need a GitHub token."
    );
  }
  if (res.status === 401) throw new GitsumError("bad_token", "GitHub rejected the token.", "Create a new token with read access to the repository.");
  if (res.status === 409) throw new GitsumError("empty", "This repository is empty.", "There are no commits to analyse yet.");
  if (res.status === 451 || res.status === 403) throw new GitsumError("blocked", "GitHub refused access to this repository.", "It may be disabled or restricted.");
  throw new GitsumError("github_error", `GitHub returned an error (${res.status}).`, "Try again in a moment.");
}

/** Repo metadata + full file tree. Two API calls, cached for 10 minutes. */
export async function loadRepo(owner, repo, opts = {}) {
  const key = `repo:${owner}/${repo}:${opts.token ? "t" : "a"}`.toLowerCase();
  const cached = cacheGet(key);
  if (cached) return cached;

  const meta = await apiGet(`/repos/${owner}/${repo}`, opts);
  const branch = meta.default_branch || "main";
  const treeRes = await apiGet(`/repos/${meta.owner.login}/${meta.name}/git/trees/${encodeURIComponent(branch)}?recursive=1`, opts);

  const files = (treeRes.tree || []).filter((n) => n.type === "blob").map((n) => ({ path: n.path, size: n.size || 0 }));

  const result = {
    owner: meta.owner.login,
    repo: meta.name,
    fullName: meta.full_name,
    branch,
    private: !!meta.private,
    description: meta.description || "",
    homepage: meta.homepage || "",
    stars: meta.stargazers_count || 0,
    forks: meta.forks_count || 0,
    openIssues: meta.open_issues_count || 0,
    license: meta.license?.spdx_id && meta.license.spdx_id !== "NOASSERTION" ? meta.license.spdx_id : meta.license?.name || "",
    topics: meta.topics || [],
    pushedAt: meta.pushed_at,
    archived: !!meta.archived,
    fork: !!meta.fork,
    htmlUrl: meta.html_url,
    truncated: !!treeRes.truncated,
    fetchedAt: new Date().toISOString(),
    files,
  };
  cacheSet(key, result);
  return result;
}

const rawCache = new Map();

/** Raw text of one file. Public repos use raw.githubusercontent.com; private repos use the contents API. */
export async function fetchRawFile(repoData, path, { token, signal, maxBytes = 400_000, fetchImpl = globalThis.fetch } = {}) {
  const { owner, repo, branch } = repoData;
  const key = `${owner}/${repo}@${branch}:${path}`;
  if (rawCache.has(key)) return rawCache.get(key);
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  const usePrivate = repoData.private && token;
  const url = usePrivate
    ? `${API}/repos/${owner}/${repo}/contents/${encodedPath}?ref=${encodeURIComponent(branch)}`
    : `${RAW}/${owner}/${repo}/${encodeURIComponent(branch)}/${encodedPath}`;
  const promise = (async () => {
    let res;
    try {
      res = await fetchImpl(url, { signal, headers: usePrivate ? headersFor(token, "application/vnd.github.raw") : undefined });
    } catch (err) {
      if (err?.name === "AbortError") throw err;
      throw new GitsumError("network", `Couldn't load ${path}.`, "Check your connection and try again.");
    }
    if (!res.ok) throw new GitsumError("file_error", `Couldn't load ${path} (${res.status}).`);
    const text = await res.text();
    return text.length > maxBytes ? text.slice(0, maxBytes) : text;
  })();
  rawCache.set(key, promise);
  if (rawCache.size > 600) rawCache.delete(rawCache.keys().next().value);
  promise.catch(() => rawCache.delete(key));
  return promise;
}

/**
 * GitHub code search inside one repo (needs a token; 10 searches/min). Returns [{ path }].
 * Used only as a fallback when local search misses.
 */
export async function codeSearch(repoData, query, { token, signal, fetchImpl = globalThis.fetch } = {}) {
  if (!token) return [];
  const q = encodeURIComponent(`${query} repo:${repoData.owner}/${repoData.repo}`);
  try {
    const res = await fetchImpl(`${API}/search/code?q=${q}&per_page=5`, { signal, headers: headersFor(token) });
    if (!res.ok) return [];
    const body = await res.json();
    return (body.items || []).map((i) => ({ path: i.path }));
  } catch {
    return [];
  }
}

export function githubFileUrl({ owner, repo, branch }, path, startLine, endLine) {
  const lines = startLine ? `#L${startLine}${endLine && endLine !== startLine ? `-L${endLine}` : ""}` : "";
  return `https://github.com/${owner}/${repo}/blob/${encodeURIComponent(branch)}/${path.split("/").map(encodeURIComponent).join("/")}${lines}`;
}
