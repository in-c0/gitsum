// Round-trip through the official MCP client over stdio, against the fake GitHub.
import test from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

const root = fileURLToPath(new URL("..", import.meta.url));

test("MCP server lists 5 read-only tools and answers each", async () => {
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["--import", `${root}tests/fixtures/fake-github.mjs`, `${root}src/cli.js`, "mcp"],
    env: { ...process.env, GITHUB_TOKEN: "", GH_TOKEN: "", GITSUM_TEAM_KEY: "" },
    stderr: "pipe",
  });
  const client = new Client({ name: "gitsum-test", version: "0.0.0" });
  await client.connect(transport);
  try {
    const { tools } = await client.listTools();
    assert.deepEqual(tools.map((t) => t.name).sort(), ["gitsum_ask", "gitsum_find", "gitsum_overview", "gitsum_read", "gitsum_tree"]);
    for (const t of tools) assert.equal(t.annotations?.readOnlyHint, true, t.name);

    const call = async (name, args) => {
      const r = await client.callTool({ name, arguments: args });
      return { text: r.content[0].text, isError: r.isError };
    };
    assert.match((await call("gitsum_overview", { repo: "acme/widget" })).text, /Tiny JSON dashboards/);
    assert.match((await call("gitsum_ask", { repo: "https://github.com/acme/widget", question: "Where is `renderDashboard` defined?" })).text, /src\/render\/dashboard\.js:2/);
    assert.match((await call("gitsum_find", { repo: "acme/widget", symbol: "SaveWidget()" })).text, /src\/core\/store\.go/);
    assert.match((await call("gitsum_read", { repo: "acme/widget", path: "src/index.js", start: 1, end: 2 })).text, /1 {2}import/);
    assert.match((await call("gitsum_tree", { repo: "acme/widget", under: "src" })).text, /^5 files under/);
    const missing = await call("gitsum_overview", { repo: "acme/missing" });
    assert.equal(missing.isError, true);
    assert.match(missing.text, /Repository not found/);
  } finally {
    await client.close();
  }
});
