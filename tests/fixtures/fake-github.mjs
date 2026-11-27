// Replaces global fetch with a tiny fake GitHub serving one synthetic repo: acme/widget.
// Used by tests (and preloaded into the MCP server child process with --import).

export const FILES = {
  "README.md": "# Widget\n\nWidget renders tiny dashboards from JSON so teams can share metrics without a backend.\n\n## Getting started\n\n```sh\nnpm install\nnpm run dev\n```\n",
  "package.json": JSON.stringify({ name: "widget", scripts: { dev: "vite", test: "vitest" }, dependencies: { react: "^18" }, devDependencies: { vite: "^5", vitest: "^1" } }),
  "src/index.js": "import { renderDashboard } from './render/dashboard.js';\nexport default function main(el, data) {\n  return renderDashboard(el, data);\n}\n",
  "src/render/dashboard.js": "// Draws the dashboard\nexport function renderDashboard(el, data) {\n  el.textContent = JSON.stringify(data);\n  return el;\n}\n",
  "src/utils/parse-config.js": "const DEFAULTS = { theme: 'auto' };\n\nexport const parseConfig = (raw) => ({ ...DEFAULTS, ...JSON.parse(raw) });\n",
  "src/server/http_server.py": "import json\n\nclass HttpServer:\n    def handle_request(self, req):\n        return json.dumps({'ok': True})\n",
  "src/core/store.go": "package core\n\ntype Store struct{}\n\nfunc (s *Store) SaveWidget(id string) error {\n\treturn nil\n}\n",
  "tests/dashboard.test.js": "import { renderDashboard } from '../src/render/dashboard.js';\ntest('renders', () => { function renderDashboard() {} });\n",
  ".github/workflows/ci.yml": "name: ci\n",
  "LICENSE": "MIT",
};

const json = (body, status = 200, headers = {}) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

export function fakeFetch(url) {
  const u = String(url);
  if (u === "https://api.github.com/repos/acme/widget")
    return json({ name: "widget", full_name: "acme/widget", owner: { login: "acme" }, default_branch: "main", description: "Tiny JSON dashboards", stargazers_count: 1234, forks_count: 56, open_issues_count: 7, license: { spdx_id: "MIT" }, topics: ["dashboard"], pushed_at: new Date().toISOString(), html_url: "https://github.com/acme/widget", private: false });
  if (u.startsWith("https://api.github.com/repos/acme/widget/git/trees/"))
    return json({ truncated: false, tree: Object.entries(FILES).map(([path, text]) => ({ path, type: "blob", size: text.length })) });
  if (u.startsWith("https://api.github.com/repos/acme/missing")) return json({ message: "Not Found" }, 404);
  if (u.startsWith("https://api.github.com/repos/acme/limited")) return json({ message: "rate" }, 403, { "x-ratelimit-remaining": "0", "x-ratelimit-reset": String(Math.floor(Date.now() / 1000) + 600) });
  const raw = u.match(/^https:\/\/raw\.githubusercontent\.com\/acme\/widget\/main\/(.+)$/);
  if (raw) {
    const path = raw[1].split("/").map(decodeURIComponent).join("/");
    return FILES[path] !== undefined ? new Response(FILES[path]) : new Response("404", { status: 404 });
  }
  return new Response("unexpected " + u, { status: 599 });
}

globalThis.fetch = async (url) => fakeFetch(url);
