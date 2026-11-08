// Builds the HUD judge fixtures from REAL data.
//  - extension/fixtures/repo.json  : loadRepo + buildOverview output for a live repo (GitHub API); file
//                                    contents are NOT embedded — the judge serves them from a local clone.
//  - extension/fixtures/large.json : a large tree from `git ls-tree -r -l` of a local clone (no API),
//                                    labelled as a clone snapshot; raw files are served from the clone.
// Usage: NODE_USE_ENV_PROXY=1 node scripts/make-hud-fixtures.mjs [owner/repo] [largeCloneDir] [smallCloneDir]
// (clone both first, e.g. git clone --depth 1 https://github.com/withastro/astro)
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadRepo } from "../src/core/github.js";
import { buildOverview } from "../src/core/analyze.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "extension", "fixtures");
const [slug = "in-c0/gitsum", clone = "/home/claude/benchrepos/astro", smallClone = "/home/claude/benchrepos/gitsum"] = process.argv.slice(2);

// Raw file reads: try raw.githubusercontent.com; if it refuses (private repo, sandbox), read the same
// file from a local clone of the same repo. Metadata + tree always come from the live API.
const cloneFetch = async (url, init) => {
  const m = String(url).match(/^https:\/\/raw\.githubusercontent\.com\/[^/]+\/[^/]+\/[^/]+\/(.+)$/);
  if (m) {
    const res = await fetch(url, init).catch(() => null);
    if (res?.ok) return res;
    const local = join(smallClone, m[1].split("/").map(decodeURIComponent).join("/"));
    if (existsSync(local)) return new Response(readFileSync(local));
    return new Response("404", { status: 404 });
  }
  return fetch(url, init);
};

// 1. Real repo through the engine.
const [owner, repo] = slug.split("/");
const repoData = await loadRepo(owner, repo);
// Label the snapshot with the branch the clone was taken from (the one file contents come from).
repoData.branch = execFileSync("git", ["-C", smallClone, "rev-parse", "--abbrev-ref", "HEAD"], { encoding: "utf8" }).trim() || repoData.branch;
const overview = await buildOverview(repoData, { fetchImpl: cloneFetch });
// No file contents are embedded (the repo may be private): the panel reads them from the local clone,
// served by the judge at rawBase. The overview is kept for reference minus README text and package.json.
const { readme, sections, pkg, ...overviewMeta } = overview;
writeFileSync(
  join(out, "repo.json"),
  JSON.stringify({
    kind: "live-snapshot",
    source: `GitHub REST API via src/core loadRepo + buildOverview (${repoData.fullName})`,
    generatedAt: new Date().toISOString(),
    repoData,
    overview: overviewMeta,
    rawBase: `fixtures/clone/${repoData.repo}/`,
    cloneDir: smallClone,
    cloneUrl: `https://github.com/${repoData.fullName}`,
    sha: execFileSync("git", ["-C", smallClone, "rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
  })
);
console.log(`repo.json: ${repoData.fullName} ${repoData.files.length} files (contents served from ${smallClone})`);

// 2. Large tree from a local clone (git ls-tree), no invented metadata.
const git = (...a) => execFileSync("git", ["-C", clone, ...a], { encoding: "utf8", maxBuffer: 64 << 20 });
const remote = git("remote", "get-url", "origin").trim();
const m = remote.match(/github\.com[/:]([^/]+)\/([^/.]+)/);
const sha = git("rev-parse", "HEAD").trim();
const pushedAt = git("log", "-1", "--format=%cI").trim();
const files = git("ls-tree", "-r", "-l", "-z", "HEAD") // -z: no quoting of unusual paths
  .split("\0")
  .filter(Boolean)
  .map((line) => {
    const [meta, path] = line.split("\t");
    const [mode, type, , size] = meta.trim().split(/\s+/);
    return { mode, type, path, size: Number(size) || 0 };
  })
  .filter((f) => f.type === "blob" && f.mode !== "120000")
  .map(({ path, size }) => ({ path, size }));
const large = {
  owner: m?.[1] || "local",
  repo: m?.[2] || basename(clone),
  fullName: `${m?.[1] || "local"}/${m?.[2] || basename(clone)}`,
  branch: "main",
  private: false,
  description: "",
  homepage: "",
  stars: null, // unknown offline: never invented
  forks: null,
  openIssues: null,
  license: "",
  topics: [],
  pushedAt,
  archived: false,
  fork: false,
  htmlUrl: `https://github.com/${m?.[1]}/${m?.[2]}`,
  truncated: false,
  fetchedAt: new Date().toISOString(),
  files,
};
writeFileSync(
  join(out, "large.json"),
  JSON.stringify({ kind: "clone-snapshot", source: `git ls-tree -r -l of a local clone of ${remote} @ ${sha.slice(0, 7)} (no API metadata)`, generatedAt: new Date().toISOString(), repoData: large, rawBase: `fixtures/clone/${large.repo}/`, cloneDir: clone, cloneUrl: large.htmlUrl, sha })
);
console.log(`large.json: ${large.fullName} ${files.length} files`);
