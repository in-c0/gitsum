// Deterministic repository analysis. Works with no AI and no token:
// ranks files, detects the stack, reads the README, and answers questions by searching the code.

import { fetchRawFile } from "./github.js";

const IGNORED_DIRS = /(^|\/)(node_modules|dist|build|out|\.next|\.nuxt|coverage|vendor|third_party|__pycache__|\.venv|venv|target|\.git|\.idea|\.vscode|bower_components|\.yarn|Pods|DerivedData)(\/|$)/;
const BINARY_EXT = /\.(png|jpe?g|gif|webp|ico|icns|bmp|tiff?|psd|svg|pdf|zip|gz|tgz|bz2|xz|7z|rar|jar|war|class|o|a|so|dylib|dll|exe|bin|wasm|woff2?|ttf|otf|eot|mp[34]|mov|avi|webm|wav|ogg|flac|db|sqlite|pyc|lockb|onnx|pt|h5|pkl|npy)$/i;
const LOCKFILES = /(^|\/)(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|bun\.lockb|Cargo\.lock|poetry\.lock|Pipfile\.lock|composer\.lock|Gemfile\.lock|go\.sum|uv\.lock)$/;

export const LANG_BY_EXT = {
  js: "JavaScript", mjs: "JavaScript", cjs: "JavaScript", jsx: "JavaScript",
  ts: "TypeScript", tsx: "TypeScript", mts: "TypeScript", cts: "TypeScript",
  py: "Python", ipynb: "Jupyter Notebook", rb: "Ruby", go: "Go", rs: "Rust", java: "Java", kt: "Kotlin", kts: "Kotlin",
  swift: "Swift", m: "Objective-C", mm: "Objective-C", c: "C", h: "C", cc: "C++", cpp: "C++", cxx: "C++", hpp: "C++",
  cs: "C#", fs: "F#", php: "PHP", scala: "Scala", ex: "Elixir", exs: "Elixir", erl: "Erlang", hs: "Haskell",
  clj: "Clojure", dart: "Dart", lua: "Lua", r: "R", jl: "Julia", zig: "Zig", nim: "Nim", sol: "Solidity",
  vue: "Vue", svelte: "Svelte", astro: "Astro", html: "HTML", css: "CSS", scss: "SCSS", sass: "SCSS", less: "Less",
  sh: "Shell", bash: "Shell", zsh: "Shell", ps1: "PowerShell", sql: "SQL", graphql: "GraphQL", proto: "Protocol Buffers",
  tf: "HCL", nix: "Nix", ml: "OCaml", elm: "Elm", pl: "Perl", groovy: "Groovy",
};

const MANIFESTS = [
  { re: /^package\.json$/, stack: "Node.js" },
  { re: /^deno\.jsonc?$/, stack: "Deno" },
  { re: /^bun\.lockb?$/, stack: "Bun" },
  { re: /^(pyproject\.toml|setup\.py|setup\.cfg|requirements\.txt|Pipfile)$/, stack: "Python" },
  { re: /^Cargo\.toml$/, stack: "Rust / Cargo" },
  { re: /^go\.mod$/, stack: "Go modules" },
  { re: /^(pom\.xml|build\.gradle(\.kts)?|settings\.gradle(\.kts)?)$/, stack: "JVM (Maven/Gradle)" },
  { re: /^Gemfile$/, stack: "Ruby / Bundler" },
  { re: /^composer\.json$/, stack: "PHP / Composer" },
  { re: /\.(csproj|sln|fsproj)$/, stack: ".NET" },
  { re: /^Package\.swift$/, stack: "Swift Package" },
  { re: /^pubspec\.yaml$/, stack: "Flutter / Dart" },
  { re: /^mix\.exs$/, stack: "Elixir / Mix" },
  { re: /^CMakeLists\.txt$/, stack: "CMake" },
  { re: /^Makefile$/, stack: "Make" },
  { re: /^(Dockerfile|docker-compose\.ya?ml|compose\.ya?ml)$/, stack: "Docker" },
  { re: /^vercel\.json$/, stack: "Vercel" },
  { re: /^netlify\.toml$/, stack: "Netlify" },
];

const JS_FRAMEWORKS = [
  ["next", "Next.js"], ["react", "React"], ["vue", "Vue"], ["svelte", "Svelte"], ["@sveltejs/kit", "SvelteKit"],
  ["nuxt", "Nuxt"], ["astro", "Astro"], ["solid-js", "Solid"], ["@angular/core", "Angular"], ["vite", "Vite"],
  ["express", "Express"], ["fastify", "Fastify"], ["hono", "Hono"], ["@nestjs/core", "NestJS"], ["koa", "Koa"],
  ["electron", "Electron"], ["react-native", "React Native"], ["expo", "Expo"], ["tailwindcss", "Tailwind CSS"],
  ["typescript", "TypeScript"], ["jest", "Jest"], ["vitest", "Vitest"], ["prisma", "Prisma"], ["drizzle-orm", "Drizzle"],
  ["three", "three.js"], ["@reduxjs/toolkit", "Redux Toolkit"], ["openai", "OpenAI SDK"], ["@anthropic-ai/sdk", "Anthropic SDK"],
  ["langchain", "LangChain"], ["webpack", "webpack"], ["esbuild", "esbuild"], ["rollup", "Rollup"],
];

const PY_FRAMEWORKS = [
  ["django", "Django"], ["flask", "Flask"], ["fastapi", "FastAPI"], ["torch", "PyTorch"], ["tensorflow", "TensorFlow"],
  ["numpy", "NumPy"], ["pandas", "pandas"], ["pytest", "pytest"], ["pydantic", "Pydantic"], ["transformers", "Transformers"],
  ["streamlit", "Streamlit"], ["langchain", "LangChain"], ["scikit-learn", "scikit-learn"], ["sqlalchemy", "SQLAlchemy"],
];

const DIR_MEANING = {
  src: "Main source code", lib: "Library code", app: "Application code / routes", pages: "Page routes",
  components: "UI components", api: "API / server handlers", server: "Server code", client: "Client code",
  cmd: "CLI entry points", internal: "Internal packages", pkg: "Packages", packages: "Monorepo packages",
  apps: "Monorepo apps", crates: "Rust crates", test: "Tests", tests: "Tests", __tests__: "Tests", spec: "Tests",
  e2e: "End-to-end tests", docs: "Documentation", doc: "Documentation", examples: "Usage examples", example: "Usage examples",
  scripts: "Build / maintenance scripts", tools: "Tooling", bin: "Executables", public: "Static assets", static: "Static assets",
  assets: "Assets", config: "Configuration", ".github": "GitHub workflows & templates", migrations: "Database migrations",
  types: "Type definitions", utils: "Utilities", benchmarks: "Benchmarks", bench: "Benchmarks", website: "Project website",
  include: "Public headers", source: "Source code", Sources: "Swift sources", Tests: "Tests",
};

const ENTRY_NAMES = /^(main|index|app|server|cli|__main__|mod|lib|program|entry|run|start|bootstrap)\.[a-z]+$/i;
const CONFIG_NAMES = /^(vite|webpack|rollup|next|nuxt|astro|svelte|tailwind|tsconfig|babel|jest|vitest|eslint|\.eslintrc|prettier|docker-compose|vercel|netlify)[.\w-]*$/i;
export const TEST_PATH = /(^|\/)(tests?|__tests__|spec|e2e)(\/|$)|\.(test|spec)\.[a-z]+$|_test\.[a-z]+$|(^|\/)test_[^/]+\.py$/i;

const extOf = (p) => {
  const m = p.match(/\.([^./]+)$/);
  return m ? m[1].toLowerCase() : "";
};
const baseOf = (p) => p.split("/").pop();
const depthOf = (p) => p.split("/").length - 1;

export const isTextual = (p) => !BINARY_EXT.test(p) && !LOCKFILES.test(p);
export const isIgnored = (p) => IGNORED_DIRS.test(p);

export function findReadme(files) {
  const candidates = files.filter((f) => /^readme(\.(md|markdown|mdx|rst|txt|org|adoc))?$/i.test(baseOf(f.path)));
  candidates.sort((a, b) => depthOf(a.path) - depthOf(b.path) || (/\.md$/i.test(b.path) ? 1 : 0) - (/\.md$/i.test(a.path) ? 1 : 0));
  return candidates[0]?.path || null;
}

export function languageBreakdown(files) {
  const bytes = {};
  let total = 0;
  for (const f of files) {
    if (isIgnored(f.path)) continue;
    const lang = LANG_BY_EXT[extOf(f.path)];
    if (!lang) continue;
    bytes[lang] = (bytes[lang] || 0) + f.size;
    total += f.size;
  }
  return Object.entries(bytes)
    .map(([name, b]) => ({ name, pct: total ? (b / total) * 100 : 0 }))
    .sort((a, b) => b.pct - a.pct)
    .filter((l) => l.pct >= 1)
    .slice(0, 6);
}

export function topLevelStructure(files) {
  const dirs = new Map();
  let rootFiles = 0;
  for (const f of files) {
    const parts = f.path.split("/");
    if (parts.length === 1) {
      rootFiles++;
      continue;
    }
    const d = parts[0];
    dirs.set(d, (dirs.get(d) || 0) + 1);
  }
  const list = [...dirs.entries()]
    .map(([name, count]) => ({ name, count, meaning: DIR_MEANING[name] || DIR_MEANING[name.toLowerCase()] || "" }))
    .sort((a, b) => (b.meaning ? 1 : 0) - (a.meaning ? 1 : 0) || b.count - a.count);
  return { dirs: list.slice(0, 10), rootFiles };
}

/** Score files by how much they tell you about the project. Returns [{ path, reason, score }] */
export function rankKeyFiles(files, limit = 8) {
  const scored = [];
  for (const f of files) {
    const { path, size } = f;
    if (isIgnored(path) || !isTextual(path)) continue;
    const base = baseOf(path);
    const depth = depthOf(path);
    let score = 0;
    let reason = "";
    if (/^readme/i.test(base)) {
      score = 100;
      reason = "Project overview";
    } else if (MANIFESTS.some((m) => m.re.test(base)) && depth <= 1) {
      const deploy = /^(Dockerfile|docker-compose|compose|vercel\.json|netlify\.toml)/.test(base);
      score = deploy ? 40 : 85;
      reason = deploy ? "Deployment config" : "Dependencies & scripts";
    } else if (ENTRY_NAMES.test(base) && LANG_BY_EXT[extOf(path)] && !/\.(css|scss|sass|less)$/i.test(base) && !/(^|\/)(assets|\.github)\//.test(path) && !TEST_PATH.test(path)) {
      score = 70;
      reason = "Likely entry point";
    } else if (/^(routes?|router|handlers?|controllers?|models?|schema|store|config|settings|constants)\.[a-z]+$/i.test(base)) {
      score = 45;
      reason = "Core wiring";
    } else if (CONFIG_NAMES.test(base) && depth === 0) {
      score = 30;
      reason = "Build / tooling config";
    } else if (/^(CONTRIBUTING|ARCHITECTURE|DESIGN|CHANGELOG)\b/i.test(base)) {
      score = /ARCHITECTURE|DESIGN/i.test(base) ? 75 : 25;
      reason = /ARCHITECTURE|DESIGN/i.test(base) ? "Architecture notes" : "Project docs";
    } else if (LANG_BY_EXT[extOf(path)]) {
      score = 20;
      reason = TEST_PATH.test(path) ? "Tests" : "Source file";
    } else continue;

    score -= depth * 7;
    if (/^(src|lib|app|cmd|server|source|Sources)\//.test(path)) score += 12;
    if (TEST_PATH.test(path)) score -= 25;
    if (/(^|\/)(examples?|docs?|scripts?|\.github)\//.test(path) && !/^readme/i.test(base)) score -= 15;
    if (size > 200_000) score -= 30;
    else if (size > 2_000 && size < 40_000 && reason === "Source file") score += 6;
    scored.push({ path, size, reason, score });
  }
  scored.sort((a, b) => b.score - a.score || a.path.length - b.path.length);
  return scored.slice(0, limit);
}

function detectStack(files, pkg, pyText) {
  const stack = new Set();
  for (const f of files) {
    if (depthOf(f.path) > 1 || isIgnored(f.path)) continue;
    const base = baseOf(f.path);
    for (const m of MANIFESTS) if (m.re.test(base)) stack.add(m.stack);
  }
  if (files.some((f) => f.path.startsWith(".github/workflows/"))) stack.add("GitHub Actions");
  if (pkg) {
    const deps = { ...pkg.dependencies, ...pkg.devDependencies, ...pkg.peerDependencies };
    for (const [dep, label] of JS_FRAMEWORKS) if (deps[dep]) stack.add(label);
  }
  if (pyText) {
    const lower = pyText.toLowerCase();
    for (const [dep, label] of PY_FRAMEWORKS) if (new RegExp(`(^|[\\s"'\\[,])${dep}([\\s"'<>=~!\\],;]|$)`, "m").test(lower)) stack.add(label);
  }
  return [...stack].slice(0, 12);
}

// ---------- README parsing ----------

function cleanMarkdownLine(line) {
  return line
    .replace(/<[^>]+>/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    .replace(/\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)/g, "")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    .replace(/[*_`]{1,3}([^*_`]+)[*_`]{1,3}/g, "$1")
    .replace(/\s+/g, " ")
    .trim();
}

export function parseReadmeSections(md) {
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const sections = [];
  let current = { title: "Introduction", level: 1, start: 1, lines: [] };
  let inFence = false;
  lines.forEach((line, i) => {
    if (/^\s*(```|~~~)/.test(line)) inFence = !inFence;
    const h = !inFence && line.match(/^(#{1,4})\s+(.+?)\s*#*\s*$/);
    const setext = !inFence && i > 0 && /^(=+|-+)\s*$/.test(line) && lines[i - 1].trim() && current.lines.length;
    if (h) {
      sections.push(current);
      current = { title: cleanMarkdownLine(h[2]), level: h[1].length, start: i + 1, lines: [] };
    } else if (setext) {
      const title = current.lines.pop();
      sections.push(current);
      current = { title: cleanMarkdownLine(title), level: line.startsWith("=") ? 1 : 2, start: i, lines: [] };
    } else {
      current.lines.push(line);
    }
  });
  sections.push(current);
  return sections
    .map((s) => ({ ...s, body: s.lines.join("\n").trim() }))
    .filter((s) => s.body || s.title !== "Introduction");
}

export function readmeIntro(md) {
  const text = md.replace(/\r\n/g, "\n").replace(/<!--[\s\S]*?-->/g, "");
  let inFence = false;
  const paras = [];
  let buf = [];
  for (const raw of text.split("\n")) {
    if (/^\s*(```|~~~)/.test(raw)) {
      inFence = !inFence;
      continue;
    }
    if (inFence) continue;
    const line = raw.trim();
    if (!line || /^#{1,6}\s/.test(line) || /^(=+|-+)$/.test(line)) {
      if (buf.length) paras.push(buf.join(" "));
      buf = [];
      continue;
    }
    if (/^(\||>?\s*\[!|[-*+]\s|\d+[.)]\s)/.test(line)) {
      if (buf.length) paras.push(buf.join(" "));
      buf = [];
      continue;
    }
    buf.push(line);
  }
  if (buf.length) paras.push(buf.join(" "));
  for (const p of paras) {
    const clean = cleanMarkdownLine(p);
    if (clean.length >= 40 && /[a-z]/i.test(clean) && !/^(table of contents|contents)\b/i.test(clean)) {
      return clean.length > 420 ? `${clean.slice(0, 417).replace(/\s+\S*$/, "")}…` : clean;
    }
  }
  return "";
}

const RUN_TITLES = /\b(quick ?start|getting started|install(ation)?|usage|setup|set up|run(ning)?|develop(ment)?|build(ing)?|how to use|local)\b/i;

export function findRunInstructions(sections, pkg) {
  for (const s of sections) {
    if (!RUN_TITLES.test(s.title)) continue;
    const fence = s.body.match(/(```|~~~)[^\n]*\n([\s\S]*?)\1/);
    if (fence && fence[2].trim()) {
      return { source: `README → ${s.title}`, code: fence[2].trim().split("\n").slice(0, 8).join("\n") };
    }
  }
  if (pkg?.scripts) {
    const order = ["dev", "start", "build", "test"];
    const picked = order.filter((k) => pkg.scripts[k]);
    if (picked.length) {
      const pm = "npm";
      return {
        source: "package.json scripts",
        code: ["npm install", ...picked.map((k) => (k === "start" || k === "test" ? `${pm} ${k}` : `${pm} run ${k}`))].join("\n"),
      };
    }
  }
  return null;
}

// ---------- health signals ----------

function repoHealth(files, readme, repoData) {
  const has = (re) => files.some((f) => re.test(f.path));
  const days = repoData.pushedAt ? Math.floor((Date.now() - new Date(repoData.pushedAt).getTime()) / 86_400_000) : null;
  const readmeWords = readme ? (readme.match(/[\p{L}\p{N}]+/gu) || []).length : 0;
  return [
    { id: "readme", label: "README", ok: readmeWords >= 150, value: readmeWords ? `${readmeWords.toLocaleString()} words` : "none" },
    { id: "tests", label: "Tests", ok: has(TEST_PATH), value: has(TEST_PATH) ? `${files.filter((f) => TEST_PATH.test(f.path)).length} files` : "none found" },
    { id: "ci", label: "CI", ok: has(/^\.github\/workflows\/|^\.circleci\/|^\.gitlab-ci\.yml$|^azure-pipelines\.yml$/), value: has(/^\.github\/workflows\//) ? "GitHub Actions" : has(/^\.circleci\/|^\.gitlab-ci|^azure-pipelines/) ? "configured" : "none found" },
    { id: "license", label: "License", ok: !!repoData.license, value: repoData.license || "none" },
    { id: "activity", label: "Activity", ok: days !== null && days <= 180, value: days === null ? "unknown" : days <= 1 ? "today" : days < 60 ? `${days}d ago` : days < 730 ? `${Math.round(days / 30)}mo ago` : `${Math.round(days / 365)}y ago` },
    { id: "docs", label: "Docs", ok: has(/^(docs?|documentation|website)\//i) || has(/\.mdx?$/i) && files.filter((f) => /\.mdx?$/i.test(f.path)).length >= 3, value: has(/^(docs?|documentation|website)\//i) ? "docs folder" : `${files.filter((f) => /\.mdx?$/i.test(f.path)).length} markdown files` },
  ];
}

// ---------- overview ----------

export async function buildOverview(repoData, opts = {}) {
  const { files } = repoData;
  const readmePath = findReadme(files);
  const rootHas = (name) => files.find((f) => f.path === name);

  const [readme, pkgText, pyText] = await Promise.all([
    readmePath ? fetchRawFile(repoData, readmePath, opts).catch(() => "") : "",
    rootHas("package.json") ? fetchRawFile(repoData, "package.json", opts).catch(() => "") : "",
    rootHas("pyproject.toml")
      ? fetchRawFile(repoData, "pyproject.toml", opts).catch(() => "")
      : rootHas("requirements.txt")
        ? fetchRawFile(repoData, "requirements.txt", opts).catch(() => "")
        : "",
  ]);

  let pkg = null;
  try {
    pkg = pkgText ? JSON.parse(pkgText) : null;
  } catch {
    pkg = null;
  }

  const sections = readme ? parseReadmeSections(readme) : [];
  const intro = readme ? readmeIntro(readme) : "";
  const meaningfulFiles = files.filter((f) => !isIgnored(f.path));

  return {
    readmePath,
    readme,
    sections,
    pkg,
    oneLiner: repoData.description || intro || (pkg?.description ?? ""),
    intro: repoData.description && intro && !intro.startsWith(repoData.description.slice(0, 30)) ? intro : "",
    stack: detectStack(files, pkg, pyText),
    languages: languageBreakdown(files),
    structure: topLevelStructure(meaningfulFiles),
    keyFiles: rankKeyFiles(files),
    run: findRunInstructions(sections, pkg),
    fileCount: meaningfulFiles.length,
    health: repoHealth(files, readme, repoData),
  };
}

// ---------- question answering without AI ----------

const STOP = new Set(
  "a an the is are was were be been do does did how what where which who why when can could should would i you we it its this that these those of in on at to for from with by and or not repo repository project code file files use used using there here about into me my your work works happen get got".split(
    " "
  )
);

export function tokenize(text) {
  // camelCase / PascalCase identifiers also count as their parts: isPlainObject → isplainobject, plain, object
  const raw = String(text).match(/[A-Za-z0-9_$]+/g) || [];
  const out = [];
  for (const word of raw) {
    const lower = word.toLowerCase();
    out.push(lower);
    const parts = word.split(/_|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])/).filter(Boolean);
    if (parts.length > 1) for (const part of parts) out.push(part.toLowerCase());
  }
  return out
    .map((t) => (t.length > 4 ? t.replace(/(ing|ed|es|s)$/, "") : t))
    .filter((t) => t.length > 1 && !STOP.has(t));
}

function chunkFile(path, text, linesPerChunk = 40) {
  const lines = text.split("\n");
  const chunks = [];
  for (let i = 0; i < lines.length; i += linesPerChunk - 8) {
    const slice = lines.slice(i, i + linesPerChunk);
    chunks.push({ path, startLine: i + 1, endLine: i + slice.length, text: slice.join("\n") });
    if (i + linesPerChunk >= lines.length) break;
  }
  return chunks;
}

/** Fetches the most informative small text files once and builds a search index. */
export async function buildSearchIndex(repoData, overview, opts = {}) {
  const { maxFiles = 18 } = opts;
  const candidates = rankKeyFiles(repoData.files, 60)
    .filter((f) => f.size > 0 && f.size < 80_000 && f.path !== overview.readmePath)
    .slice(0, maxFiles);
  const texts = await Promise.all(
    candidates.map((f) => fetchRawFile(repoData, f.path, opts).then((t) => ({ path: f.path, text: t })).catch(() => null))
  );
  const chunks = [];
  if (overview.readme) {
    for (const s of overview.sections) {
      const body = `${s.title}\n${s.body}`;
      chunks.push({ path: overview.readmePath, startLine: s.start, endLine: s.start + s.lines.length, text: body, title: s.title });
    }
  }
  for (const t of texts) if (t) chunks.push(...chunkFile(t.path, t.text));

  const docs = chunks.map((c) => {
    const toks = tokenize(`${c.path.replace(/[/.-]/g, " ")} ${c.title || ""} ${c.text}`);
    const tf = new Map();
    for (const tk of toks) tf.set(tk, (tf.get(tk) || 0) + 1);
    return { ...c, tf, len: toks.length };
  });
  const df = new Map();
  for (const d of docs) for (const tk of d.tf.keys()) df.set(tk, (df.get(tk) || 0) + 1);
  const avgLen = docs.reduce((s, d) => s + d.len, 0) / (docs.length || 1);
  return { docs, df, avgLen, pathList: repoData.files.map((f) => f.path) };
}

const escapeRe = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Identifiers the user typed (isPlainObject, parse_args, UserService) — used to find where they are defined. */
export function queryIdentifiers(query) {
  return [...new Set((query.match(/[A-Za-z_$][A-Za-z0-9_$]{2,}/g) || []).filter((w) => !STOP.has(w.toLowerCase()) && (/[A-Z_$]/.test(w.slice(1)) || /\(\)$/.test(w))))];
}

export function definitionLine(text, identifiers) {
  if (!identifiers.length) return -1;
  const re = new RegExp(
    `(^|\\s)(export\\s+)?(default\\s+)?(async\\s+)?(function\\*?|def|class|interface|type|struct|enum|trait|fn|func|const|let|var|val|public|private|static)\\s+(${identifiers.map(escapeRe).join("|")})\\b`,
    "i"
  );
  return text.split("\n").findIndex((line) => re.test(line));
}

export function searchIndex(index, query, k = 3) {
  const q = [...new Set(tokenize(query))];
  if (!q.length) return [];
  const idents = queryIdentifiers(query);
  const N = index.docs.length;
  const k1 = 1.4;
  const b = 0.75;
  const results = index.docs
    .map((d) => {
      let score = 0;
      for (const t of q) {
        const f = d.tf.get(t);
        if (!f) continue;
        const idf = Math.log(1 + (N - (index.df.get(t) || 0) + 0.5) / ((index.df.get(t) || 0) + 0.5));
        score += idf * ((f * (k1 + 1)) / (f + k1 * (1 - b + (b * d.len) / index.avgLen)));
      }
      if (q.some((t) => d.path.toLowerCase().includes(t))) score *= 1.3;
      if (TEST_PATH.test(d.path)) score *= 0.6; // prefer the implementation over its tests
      const def = score > 0 ? definitionLine(d.text, idents) : -1;
      if (def >= 0) score *= 3; // where it is defined beats where it is used
      return { d, score, def };
    })
    .filter((r) => r.score > 0.5)
    .sort((a, b) => b.score - a.score);

  const picked = [];
  for (const r of results) {
    if (picked.some((p) => p.d.path === r.d.path && Math.abs(p.d.startLine - r.d.startLine) < 40)) continue;
    picked.push(r);
    if (picked.length >= k) break;
  }
  return picked.map(({ d, def }) => ({ path: d.path, startLine: d.startLine, endLine: d.endLine, title: d.title, snippet: bestSnippet(d.text, q, def) }));
}

function bestSnippet(text, qTokens, preferLine = -1, size = 10) {
  const lines = text.split("\n");
  let best = 0;
  if (preferLine >= 0) {
    best = Math.max(0, preferLine - 1);
  } else {
    let bestScore = -1;
    for (let i = 0; i < lines.length; i++) {
      const window = lines.slice(i, i + size).join(" ").toLowerCase();
      const s = qTokens.reduce((acc, t) => acc + (window.includes(t) ? 1 : 0), 0);
      if (s > bestScore) {
        bestScore = s;
        best = i;
      }
    }
  }
  while (best < lines.length - 1 && !lines[best].trim()) best++;
  const out = lines.slice(best, best + size);
  return { text: out.join("\n").trimEnd(), offset: best };
}

/** Answers common questions from the overview; everything else by searching the code. */
export function answerLocally(question, repoData, overview, index) {
  const q = question.toLowerCase();
  const cite = (path, extra = {}) => ({ path, ...extra });

  if (/\b(what (is|does)|purpose|about|overview|summar|explain (this|the) (repo|project))\b/.test(q) && !/\b(file|function|class|folder|directory)\b/.test(q)) {
    const parts = [overview.oneLiner, overview.intro].filter(Boolean);
    if (parts.length) {
      return { text: parts.join("\n\n"), citations: overview.readmePath ? [cite(overview.readmePath)] : [], kind: "overview" };
    }
  }
  if (/\b(entry ?point|main file|where (do|should) i (start|begin)|start(ing)? (reading|point)|bootstrap)\b/.test(q)) {
    const entries = overview.keyFiles.filter((f) => ["Likely entry point", "Dependencies & scripts", "Architecture notes"].includes(f.reason));
    if (entries.length) return { text: "Start with these files:", citations: entries.slice(0, 5).map((f) => cite(f.path, { note: f.reason })), kind: "files" };
  }
  if (/\b(run|install|set ?up|start|build|launch|usage|get started|getting started|use it)\b/.test(q) && !/\btests?\b/.test(q) && overview.run) {
    return { text: `From ${overview.run.source}:`, code: overview.run.code, citations: [cite(overview.run.source.startsWith("README") ? overview.readmePath : "package.json")], kind: "run" };
  }
  if (/\b(stack|language|framework|built with|tech|dependenc|librar)\w*\b/.test(q) && (overview.stack.length || overview.languages.length)) {
    const langs = overview.languages.map((l) => `${l.name} ${Math.round(l.pct)}%`).join(", ");
    const text = [overview.stack.length ? `Stack: ${overview.stack.join(", ")}.` : "", langs ? `Languages by size: ${langs}.` : ""].filter(Boolean).join("\n");
    const citations = ["package.json", "pyproject.toml", "requirements.txt", "Cargo.toml", "go.mod"].filter((p) => index.pathList.includes(p)).map((p) => cite(p));
    return { text, citations, kind: "stack" };
  }
  if (/\b(structure|organi[sz]ed|folders?|director(y|ies)|layout|architecture)\b/.test(q)) {
    const lines = overview.structure.dirs.map((d) => `${d.name}/ — ${d.meaning || `${d.count} files`}`);
    if (lines.length) return { text: lines.join("\n"), citations: overview.keyFiles.slice(0, 3).map((f) => cite(f.path, { note: f.reason })), kind: "structure" };
  }
  if (/\blicen[cs]e\b/.test(q)) {
    const lic = index.pathList.find((p) => /^licen[cs]e/i.test(p));
    return { text: repoData.license ? `Licensed under ${repoData.license}.` : "No license detected by GitHub.", citations: lic ? [cite(lic)] : [], kind: "license" };
  }

  const hits = searchIndex(index, question);
  const qTokens = tokenize(question);
  const pathHits = index.pathList.filter((p) => qTokens.some((t) => t.length > 2 && baseOf(p).toLowerCase().includes(t))).slice(0, 4);

  if (!hits.length && !pathHits.length) {
    return { text: "I couldn't find anything in the key files that matches. Try naming a file, function, or feature.", citations: [], kind: "none" };
  }
  return {
    text: hits.length ? "Most relevant places in the code:" : "Files whose names match your question:",
    citations: [
      ...hits.map((h) => ({
        path: h.path,
        startLine: h.startLine + h.snippet.offset,
        endLine: h.startLine + h.snippet.offset + h.snippet.text.split("\n").length - 1,
        note: h.title,
        snippet: h.snippet.text,
      })),
      ...pathHits.filter((p) => !hits.some((h) => h.path === p)).map((p) => ({ path: p })),
    ],
    kind: "search",
  };
}

export function suggestedQuestions(overview) {
  const qs = ["What does this project do?"];
  if (overview.run) qs.push("How do I run it locally?");
  qs.push("Where should I start reading the code?");
  if (overview.stack.length) qs.push("What's the tech stack?");
  else qs.push("How is the code organised?");
  return qs.slice(0, 4);
}
