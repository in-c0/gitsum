#!/usr/bin/env node
// gitsum — understand any GitHub repo from the terminal or from an agent.
//   npx -y github:in-c0/gitsum#main overview expressjs/express
//   npx -y github:in-c0/gitsum#main ask expressjs/express "Where is the router defined?"
//   npx -y github:in-c0/gitsum#main mcp            # MCP server over stdio

import { ask, listTree, openRepo, readFile } from "./core/engine.js";
import { huntDefinition } from "./core/find.js";
import { githubFileUrl } from "./core/github.js";
import { formatAnswer, formatFile, formatOverview, formatTree } from "./core/format.js";
import { teamAsk, teamBody, teamConfig } from "./core/team.js";

const HELP = `gitsum — understand any GitHub repo in seconds (free for individuals)

Usage
  gitsum overview <repo>                 what it is, stack, health, how to run, files to read first
  gitsum ask <repo> "<question>"         answer with path:line evidence
  gitsum find <repo> <symbol>            where a function/class/type is defined (whole tree)
  gitsum read <repo> <path> [start] [end]  file with line numbers
  gitsum tree <repo> [folder] [--match s]  list files
  gitsum mcp                             run the MCP server over stdio
  gitsum --json …                        machine-readable output

<repo> is owner/repo or any github.com URL.
Env: GITHUB_TOKEN (private repos, higher limits) · GITSUM_TEAM_KEY + GITSUM_API (Team plan AI answers)`;

const argv = process.argv.slice(2);
const json = argv.includes("--json");
const args = argv.filter((a) => a !== "--json");
const matchIdx = args.indexOf("--match");
const match = matchIdx >= 0 ? args.splice(matchIdx, 2)[1] : "";
const [cmd, repo, ...rest] = args;
const token = process.env.GITHUB_TOKEN || process.env.GH_TOKEN || undefined;

function out(value, markdown) {
  process.stdout.write(`${json ? JSON.stringify(value, null, 2) : markdown}\n`);
}

async function main() {
  if (!cmd || cmd === "help" || cmd === "--help" || cmd === "-h") return out({ help: HELP }, HELP);
  if (cmd === "--version" || cmd === "-v") return out({ version: "0.1.0" }, "0.1.0");
  if (cmd === "mcp") {
    const { startStdio } = await import("./mcp.js");
    return startStdio();
  }
  if (!repo) throw Object.assign(new Error("Missing <repo>."), { hint: "Example: gitsum overview expressjs/express" });
  const session = await openRepo(repo, { token });

  if (cmd === "overview") {
    const { repoData, overview } = session;
    const { readme, sections, pkg, ...rest2 } = overview;
    return out({ repo: { ...repoData, files: undefined, fileCount: repoData.files.length }, overview: rest2 }, formatOverview(session));
  }
  if (cmd === "ask") {
    const question = rest.join(" ").trim();
    if (!question) throw Object.assign(new Error("Missing question."), { hint: 'Example: gitsum ask o/r "How do I run it?"' });
    const answer = await ask(session, question);
    let md = formatAnswer(question, answer);
    const team = teamConfig();
    let ai = null;
    if (team.enabled) {
      ai = await teamAsk(team, teamBody(session, question, answer.citations)).catch((err) => ({ error: err.message }));
      md = ai.error ? `${md}\n\n(Team answer unavailable: ${ai.error})` : `${ai.text}\n\n---\n${md}`;
    }
    return out({ answer, ai }, md);
  }
  if (cmd === "find") {
    const symbol = (rest[0] || "").replace(/\(\)$/, "");
    if (!symbol) throw Object.assign(new Error("Missing symbol."), { hint: "Example: gitsum find o/r parseArgs" });
    const defs = await huntDefinition(session.repoData, session.index, [symbol], session.opts);
    const citations = defs.map((d) => ({ ...d, note: `definition (${d.via})`, url: githubFileUrl(session.repoData, d.path, d.startLine, d.endLine) }));
    const answer = { text: defs.length ? `\`${symbol}\` is defined in ${defs[0].path}:${defs[0].line}.` : `No definition of \`${symbol}\` found.`, citations };
    return out(answer, formatAnswer(`Where is \`${symbol}\` defined?`, answer));
  }
  if (cmd === "read") {
    const [path, start, end] = rest;
    const f = await readFile(session, path, { start: start ? Number(start) : 1, end: end ? Number(end) : undefined });
    return out(f, formatFile(f));
  }
  if (cmd === "tree") {
    const t = listTree(session, { under: rest[0] || "", match });
    return out(t, formatTree(t, rest[0]));
  }
  throw Object.assign(new Error(`Unknown command: ${cmd}`), { hint: "Run gitsum --help" });
}

main().catch((err) => {
  process.stderr.write(`gitsum: ${err.message}${err.hint ? `\n${err.hint}` : ""}\n`);
  process.exit(1);
});
