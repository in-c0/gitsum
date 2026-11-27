// Gitsum MCP server (stdio). Free for individuals: every tool returns evidence the calling model can cite.
//   claude mcp add gitsum -- npx -y github:in-c0/gitsum mcp
// Env: GITHUB_TOKEN (optional: private repos, 5,000 req/h), GITSUM_TEAM_KEY + GITSUM_API (optional: hosted Team answers)

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { ask, listTree, openRepo, readFile } from "./core/engine.js";
import { huntDefinition } from "./core/find.js";
import { githubFileUrl } from "./core/github.js";
import { formatAnswer, formatFile, formatOverview, formatTree } from "./core/format.js";
import { teamAsk, teamBody, teamConfig } from "./core/team.js";

export const VERSION = "0.1.0";

const token = () => process.env.GITHUB_TOKEN || process.env.GH_TOKEN || undefined;
const repoArg = z.string().min(3).describe("GitHub repository: owner/repo or any github.com URL");

const text = (t) => ({ content: [{ type: "text", text: t }] });
const failure = (err) => ({
  isError: true,
  content: [{ type: "text", text: `${err.message}${err.hint ? `\n${err.hint}` : ""}` }],
});

async function guarded(fn) {
  try {
    return await fn();
  } catch (err) {
    return failure(err);
  }
}

const READ_ONLY = { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: true };

export function createServer() {
  const server = new McpServer({ name: "gitsum", version: VERSION });

  server.registerTool(
    "gitsum_overview",
    {
      title: "Repository overview",
      description:
        "Start here for any unfamiliar GitHub repo. Returns what it is, stack, languages, health, how to run it, folder meanings and the files to read first — in one compact call instead of crawling the tree.",
      inputSchema: { repo: repoArg },
      annotations: READ_ONLY,
    },
    ({ repo }) => guarded(async () => text(formatOverview(await openRepo(repo, { token: token() }))))
  );

  server.registerTool(
    "gitsum_ask",
    {
      title: "Ask a repository",
      description:
        "Answer a question about a GitHub repo with evidence: exact path:line citations and code snippets (definitions, run commands, structure, keyword search over key files). Use the citations to answer; read more lines with gitsum_read if needed.",
      inputSchema: { repo: repoArg, question: z.string().min(2).max(2000).describe("The question, e.g. 'Where is `createServer` defined?' or 'How is auth handled?'") },
      annotations: READ_ONLY,
    },
    ({ repo, question }) =>
      guarded(async () => {
        const session = await openRepo(repo, { token: token() });
        const answer = await ask(session, question);
        let out = formatAnswer(question, answer);
        const team = teamConfig();
        if (team.enabled) {
          try {
            const ai = await teamAsk(team, teamBody(session, question, answer.citations));
            out = `${ai.text}\n\n---\nEvidence:\n${out}`;
          } catch (err) {
            out += `\n\n(Team answer unavailable: ${err.message})`;
          }
        }
        return text(out);
      })
  );

  server.registerTool(
    "gitsum_find",
    {
      title: "Find a definition",
      description: "Find where a function, class, type or constant is defined in a GitHub repo, searching the whole tree (not just key files). Returns path:line and the definition snippet.",
      inputSchema: { repo: repoArg, symbol: z.string().min(2).max(120).describe("Identifier, e.g. parseArgs, UserService, MAX_RETRIES") },
      annotations: READ_ONLY,
    },
    ({ repo, symbol }) =>
      guarded(async () => {
        const session = await openRepo(repo, { token: token() });
        const clean = symbol.replace(/\(\)$/, "").split(/[.#:]/).pop();
        const defs = await huntDefinition(session.repoData, session.index, [clean], session.opts);
        if (!defs.length) return text(`No definition of \`${clean}\` found in ${session.repoData.fullName} (searched key files, files named like it, and up to 80 source files). Try gitsum_ask with a description instead.`);
        const answer = {
          text: `\`${clean}\` is defined in ${defs[0].path}:${defs[0].line}.`,
          citations: defs.map((d) => ({ ...d, note: `definition (${d.via})`, url: githubFileUrl(session.repoData, d.path, d.startLine, d.endLine) })),
        };
        return text(formatAnswer(`Where is \`${clean}\` defined?`, answer));
      })
  );

  server.registerTool(
    "gitsum_read",
    {
      title: "Read a file",
      description: "Read a file from a GitHub repo with line numbers (up to 220 lines per call). Use start/end to page.",
      inputSchema: {
        repo: repoArg,
        path: z.string().min(1).describe("File path inside the repo"),
        start: z.number().int().min(1).optional().describe("First line (1-based)"),
        end: z.number().int().min(1).optional().describe("Last line"),
      },
      annotations: READ_ONLY,
    },
    ({ repo, path, start, end }) => guarded(async () => text(formatFile(await readFile(await openRepo(repo, { token: token() }), path, { start, end }))))
  );

  server.registerTool(
    "gitsum_tree",
    {
      title: "List files",
      description: "List files in a GitHub repo, optionally under a folder or matching a substring (max 200 per call).",
      inputSchema: {
        repo: repoArg,
        under: z.string().optional().describe("Folder, e.g. src/server"),
        match: z.string().optional().describe("Substring filter, e.g. auth or .test."),
      },
      annotations: READ_ONLY,
    },
    ({ repo, under, match }) => guarded(async () => text(formatTree(listTree(await openRepo(repo, { token: token() }), { under, match }), under)))
  );

  return server;
}

export async function startStdio() {
  const server = createServer();
  await server.connect(new StdioServerTransport());
}
