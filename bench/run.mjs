#!/usr/bin/env node
// Outside judge for Gitsum's answers. Not graded by Gitsum: questions and answers come from real repos.
//
// For each repo: shallow-clone it, sample identifiers that are defined in exactly one non-test source
// file (seeded, reproducible), ask "Where is `X` defined?", and score whether the true file is in the
// top-3 citations. File contents are served from the clone so the run is deterministic and offline
// after the first clone. Also reports latency and tokens returned vs a naive README+tree dump.
//
//   node bench/run.mjs [--repos a/b,c/d] [--per 6] [--seed 7] [--min 0.8]

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ask, openRepo } from "../src/core/engine.js";
import { formatAnswer } from "../src/core/format.js";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : fallback;
};

const REPOS = arg(
  "repos",
  "expressjs/express,pallets/flask,psf/requests,sindresorhus/ky,axios/axios,spf13/cobra,gin-gonic/gin,BurntSushi/ripgrep,colinhacks/zod,tj/commander.js"
).split(",");
const PER = Number(arg("per", 6));
const SEED = Number(arg("seed", 7));
const MIN = Number(arg("min", 0));
const CACHE = arg("cache", join(process.cwd(), ".bench-cache"));

const DEF_PATTERNS = [
  { ext: /\.(m?js|cjs|jsx|ts|tsx)$/, re: /^(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?|class)\s+([A-Za-z_$][\w$]{4,})/gm },
  { ext: /\.py$/, re: /^(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w{4,})/gm },
  { ext: /\.go$/, re: /^func\s+(?:\([^)]*\)\s+)?([A-Za-z_]\w{4,})\s*[([]/gm },
  { ext: /\.rs$/, re: /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:fn|struct|enum|trait)\s+([A-Za-z_]\w{4,})/gm },
];
const SKIP = /(^|\/)(testdata|tests?|__tests__|spec|e2e|examples?|docs?|benchmarks?|bench|fixtures?|scripts?|vendor|node_modules|dist|build|types?|typings)(\/|$)|\.(test|spec|d)\.[a-z]+$|_test\.go$/i;
const COMMON = new Set("__init__ constructor toString render handle create update delete error setup close start build parse write print value index main default helper wrapper request response handler string number object".split(" "));

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 2 ** 32);
}

function clone(full) {
  const dir = join(CACHE, full.replace("/", "__"));
  if (!existsSync(join(dir, ".git"))) {
    mkdirSync(CACHE, { recursive: true });
    execFileSync("git", ["clone", "-q", "--depth", "1", `https://github.com/${full}`, dir], { stdio: "inherit", timeout: 600_000 });
  }
  return dir;
}

function repoDataFrom(full, dir) {
  const branch = execFileSync("git", ["-C", dir, "rev-parse", "--abbrev-ref", "HEAD"]).toString().trim();
  const lines = execFileSync("git", ["-C", dir, "ls-tree", "-r", "-l", "HEAD"], { maxBuffer: 1 << 28 }).toString().trim().split("\n");
  const files = lines
    .map((l) => l.match(/^\d+\s+blob\s+\w+\s+(\d+|-)\t(.+)$/))
    .filter(Boolean)
    .map((m) => ({ path: m[2], size: Number(m[1]) || 0 }));
  const [owner, repo] = full.split("/");
  return { owner, repo, fullName: full, branch, private: false, description: "", homepage: "", stars: 0, forks: 0, openIssues: 0, license: "", topics: [], pushedAt: new Date().toISOString(), archived: false, fork: false, htmlUrl: `https://github.com/${full}`, truncated: false, files };
}

function localFetch(dir, branch, full) {
  const prefix = `https://raw.githubusercontent.com/${full}/${encodeURIComponent(branch)}/`;
  return async (url) => {
    if (!url.startsWith(prefix)) return new Response("not local", { status: 404 });
    const path = url.slice(prefix.length).split("/").map(decodeURIComponent).join("/");
    try {
      return new Response(readFileSync(join(dir, path), "utf8"), { status: 200 });
    } catch {
      return new Response("missing", { status: 404 });
    }
  };
}

function sampleQuestions(dir, files, rand) {
  const defs = new Map(); // ident -> Set(paths)
  for (const f of files) {
    if (SKIP.test(f.path) || f.size > 300_000) continue;
    const pat = DEF_PATTERNS.find((p) => p.ext.test(f.path));
    if (!pat) continue;
    let text;
    try {
      text = readFileSync(join(dir, f.path), "utf8");
    } catch {
      continue;
    }
    for (const m of text.matchAll(pat.re)) {
      const id = m[1];
      if (COMMON.has(id.toLowerCase())) continue;
      if (!defs.has(id)) defs.set(id, new Set());
      defs.get(id).add(f.path);
    }
  }
  const unique = [...defs.entries()].filter(([, paths]) => paths.size === 1).map(([id, paths]) => ({ id, truth: [...paths][0] }));
  unique.sort((a, b) => (a.id < b.id ? -1 : 1));
  const picked = [];
  while (picked.length < PER && unique.length) picked.push(unique.splice(Math.floor(rand() * unique.length), 1)[0]);
  return picked;
}

const tokens = (s) => Math.ceil(s.length / 4);

const rows = [];
const rand = rng(SEED);
for (const full of REPOS) {
  const dir = clone(full);
  const repoData = repoDataFrom(full, dir);
  const fetchImpl = localFetch(dir, repoData.branch, full);
  const t0 = performance.now();
  const session = await openRepo(full, { repoData, fetchImpl });
  const openMs = performance.now() - t0;
  const readme = session.overview.readme || "";
  const naive = tokens(readme + repoData.files.map((f) => f.path).join("\n"));
  const qs = sampleQuestions(dir, repoData.files, rand);
  qs.forEach((q, i) => {
    q.question = i % 2 ? `where is ${q.id} defined?` : `Where is \`${q.id}\` defined?`;
  });
  for (const q of qs) {
    const t = performance.now();
    const a = await ask(session, q.question, { fetchImpl });
    const ms = performance.now() - t;
    const paths = (a.citations || []).map((c) => c.path);
    rows.push({ repo: full, files: repoData.files.length, q: q.question, truth: q.truth, top: paths.slice(0, 3), hit1: paths[0] === q.truth, hit3: paths.slice(0, 3).includes(q.truth), ms: Math.round(ms), openMs: Math.round(openMs), tokens: tokens(formatAnswer(q.question, a)), naive });
  }
  const mine = rows.filter((r) => r.repo === full);
  console.log(`${full.padEnd(24)} files ${String(repoData.files.length).padStart(5)}  hit@3 ${mine.filter((r) => r.hit3).length}/${mine.length}  open ${Math.round(openMs)}ms`);
}

const n = rows.length;
const hit3 = rows.filter((r) => r.hit3).length / n;
const hit1 = rows.filter((r) => r.hit1).length / n;
const med = (xs) => xs.sort((a, b) => a - b)[Math.floor(xs.length / 2)];
const summary = {
  generatedAt: new Date().toISOString(),
  seed: SEED,
  questions: n,
  hit1: Number(hit1.toFixed(3)),
  hit3: Number(hit3.toFixed(3)),
  medianAskMs: med(rows.map((r) => r.ms)),
  p95AskMs: rows.map((r) => r.ms).sort((a, b) => a - b)[Math.floor(n * 0.95) - 1] ?? null,
  medianTokens: med(rows.map((r) => r.tokens)),
  maxTokens: Math.max(...rows.map((r) => r.tokens)),
  medianNaiveTokens: med([...new Set(rows.map((r) => r.naive))]),
};
console.log("\n", summary);
for (const r of rows.filter((x) => !x.hit3)) console.log(`  miss ${r.repo}: ${r.q} → truth ${r.truth}; got ${r.top.join(", ") || "nothing"}`);
mkdirSync("bench", { recursive: true });
writeFileSync("bench/results.json", JSON.stringify({ summary, rows }, null, 2));
if (hit3 < MIN) {
  console.error(`hit@3 ${hit3.toFixed(3)} is below --min ${MIN}`);
  process.exit(1);
}
