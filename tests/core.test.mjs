import "./fixtures/fake-github.mjs";
import test from "node:test";
import assert from "node:assert/strict";
import { parseRepoInput } from "../src/core/github.js";
import { ask, listTree, openRepo, readFile } from "../src/core/engine.js";
import { definitionMatches, questionIdentifiers } from "../src/core/find.js";
import { formatAnswer, formatOverview } from "../src/core/format.js";

test("parses every common repo spelling and rejects product pages", () => {
  for (const s of ["acme/widget", "https://github.com/acme/widget", "github.com/acme/widget/tree/main/src", "git@github.com:acme/widget.git", "https://gitsum.example/acme/widget"])
    assert.deepEqual(parseRepoInput(s), { owner: "acme", repo: "widget" }, s);
  for (const s of ["", "acme", "https://github.com/settings/profile", "https://github.com/orgs/acme"]) assert.equal(parseRepoInput(s), null, s);
});

test("question identifiers: backticks, camelCase, and 'where is X defined'", () => {
  assert.deepEqual(questionIdentifiers("Where is `renderDashboard` defined?"), ["renderDashboard"]);
  assert.ok(questionIdentifiers("where is delay defined?").includes("delay"));
  assert.ok(questionIdentifiers("how does parseConfig work").includes("parseConfig"));
});

test("definition matcher across languages prefers real definitions over calls", () => {
  const cases = [
    ["export default async function delay(ms) {", "delay"],
    ["class HttpServer:", "HttpServer"],
    ["func (s *Store) SaveWidget(id string) error {", "SaveWidget"],
    ["pub fn build_index(x: u8) -> u8 {", "build_index"],
    ["export const parseConfig = (raw) => raw;", "parseConfig"],
    ["module.exports.send = function () {}", "send"],
  ];
  for (const [line, id] of cases) assert.equal(definitionMatches(line, [id]).length, 1, line);
  for (const line of ["  return renderDashboard(el)", "        raise HttpServer(", "import { parseConfig } from './x'"]) assert.equal(definitionMatches(line, ["renderDashboard", "HttpServer", "parseConfig"]).length, 0, line);
});

test("overview: stack, run command, health and key files from the real tree", async () => {
  const s = await openRepo("acme/widget");
  assert.equal(s.repoData.files.length, 10);
  assert.ok(s.overview.stack.includes("React"));
  assert.match(s.overview.run.code, /npm run dev/);
  const md = formatOverview(s);
  assert.match(md, /# acme\/widget/);
  assert.match(md, /Read first/);
});

test("definition hunt finds files outside the key-file index and skips tests", async () => {
  const s = await openRepo("acme/widget");
  for (const [q, path] of [
    ["Where is `renderDashboard` defined?", "src/render/dashboard.js"],
    ["where is parseConfig defined?", "src/utils/parse-config.js"],
    ["Where is `SaveWidget` defined?", "src/core/store.go"],
    ["Where is `HttpServer` defined?", "src/server/http_server.py"],
  ]) {
    const a = await ask(s, q);
    assert.equal(a.kind, "definition", q);
    assert.equal(a.citations[0].path, path, q);
    assert.match(a.citations[0].url, /^https:\/\/github\.com\/acme\/widget\/blob\/main\//);
  }
});

test("run and stack questions answer from the overview", async () => {
  const s = await openRepo("acme/widget");
  assert.equal((await ask(s, "How do I run it?")).kind, "run");
  assert.equal((await ask(s, "What's the tech stack?")).kind, "stack");
});

test("read and tree", async () => {
  const s = await openRepo("acme/widget");
  const f = await readFile(s, "src/index.js", { start: 2, end: 3 });
  assert.equal(f.startLine, 2);
  assert.match(f.text, /export default function main/);
  await assert.rejects(readFile(s, "src/indx.js"), (e) => e.code === "no_file" && /src\/index\.js/.test(e.hint));
  assert.equal(listTree(s, { under: "src" }).total, 5);
});

test("errors carry a code and a useful hint", async () => {
  await assert.rejects(openRepo("acme/missing"), (e) => e.code === "not_found" && /token/.test(e.hint));
  await assert.rejects(openRepo("acme/limited"), (e) => e.code === "rate_limited" && /min/.test(e.hint));
  await assert.rejects(openRepo("not a repo"), (e) => e.code === "bad_input");
});

test("answers stay under the 4k-token budget", async () => {
  const s = await openRepo("acme/widget");
  const a = await ask(s, "dashboard render json");
  assert.ok(formatAnswer("q", a).length < 16_000);
});
