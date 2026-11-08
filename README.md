<div align="center">

# gitsum

### Understand any GitHub repo in seconds — wherever you already are.

Gitsum reads a repository for you — what it is, how to run it, where things live — and answers with exact `path:line` evidence.
One engine, three front doors: **Chrome side panel**, **MCP server**, **Agent Skill** (plus a CLI).

**Free for individuals. Team plan for companies.**

[![CI](https://github.com/in-c0/gitsum/actions/workflows/ci.yml/badge.svg)](https://github.com/in-c0/gitsum/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/hit%403-98%25-2f8a84)](#benchmark)
[![MCP](https://img.shields.io/badge/MCP-5%20read--only%20tools-c98a1b)](#mcp-server)

[Website](https://in-c0.github.io/gitsum/) · [Download extension](https://github.com/in-c0/gitsum/releases/latest/download/gitsum-extension.zip) · [Pricing](#pricing) · [Issues](https://github.com/in-c0/gitsum/issues)

</div>

## Quick start 🚀

### Let your agent do it

Paste this into any agent that can read a Skill and use a terminal (Claude Code, Cursor, Codex, …):

```text
Use this Skill: https://raw.githubusercontent.com/in-c0/gitsum/main/skills/gitsum/SKILL.md
Explain how expressjs/express routes a request, with file:line citations.
```

### Chrome extension

1. [Download the latest build](https://github.com/in-c0/gitsum/releases/latest/download/gitsum-extension.zip) and unzip it.
2. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked**, pick the unzipped folder.
3. Open any GitHub repo and click **Gitsum** — the side panel draws the repo as strata (folders as layers, files as grains coloured by language), then lets you ask.

### MCP server

```sh
claude mcp add gitsum -- npx -y github:in-c0/gitsum#main mcp
```

Any other client (`mcp.json`):

```json
{ "mcpServers": { "gitsum": { "command": "npx", "args": ["-y", "github:in-c0/gitsum#main", "mcp"] } } }
```

| Tool | What it returns |
|---|---|
| `gitsum_overview` | Purpose, stack, languages, health, run command, folder meanings, files to read first |
| `gitsum_ask` | Evidence for a question: `path:line` citations + snippets |
| `gitsum_find` | Where a function / class / type is defined — searches the whole tree, not just key files |
| `gitsum_read` | A file with line numbers (220 lines per call) |
| `gitsum_tree` | Files under a folder or matching a substring |

All tools are read-only. The calling model writes the answer from Gitsum's evidence, so no AI key is needed.

### Claude Code plugin (Skill + MCP together)

```text
/plugin marketplace add in-c0/gitsum
/plugin install gitsum@gitsum
```

### CLI

```sh
npx -y github:in-c0/gitsum#main overview pallets/flask
npx -y github:in-c0/gitsum#main ask pallets/flask "How do I run it?"
npx -y github:in-c0/gitsum#main find pallets/flask url_for
npx -y github:in-c0/gitsum#main read pallets/flask src/flask/app.py 1 60
```

Add `--json` for machine-readable output.

## Features 🎯

- [x] **Overview in one call** — replaces "read the README, list the tree, open package.json" with ~150–700 tokens.
- [x] **Definition hunt** — finds `X` across the whole tree: indexed key files → files named like `X` → GitHub code search (with a token) → a bounded sweep of source files. Prefers implementations over `.d.ts` and tests.
- [x] **Evidence, not vibes** — every answer cites `path:line` with a GitHub link.
- [x] **Private repos** — set `GITHUB_TOKEN` (also raises GitHub's limit from 60 to 5,000 requests/hour).
- [x] **Living-data side panel** — one screen, no scroll: strata of the real tree, matching files light up as you type, run command and health readouts.
- [x] **Bring your own key** — AI answers in the extension with your Anthropic key, stored only in your browser.

## Benchmark

An outside judge, not Gitsum grading itself: [`bench/run.mjs`](bench/run.mjs) shallow-clones 10 popular repos (express, flask, requests, ky, axios, cobra, gin, ripgrep, zod, commander), samples identifiers defined in exactly one non-test source file, asks *"Where is `X` defined?"* and checks whether the true file is in the top 3 citations.

| Run | Questions | hit@1 | hit@3 | Median tokens returned |
|---|---|---|---|---|
| Seed 11 (held out) | 80 | 97.5% | 97.5% | 153 |
| Seed 23 (held out) | 80 | 97.5% | 97.5% | 149 |
| Seed 42 (held out) | 80 | 98.8% | 98.8% | 160 |

Reading the README plus the file list instead costs a median of ~7,100 tokens. The remaining misses are names defined twice (e.g. zod v3 and v4). Re-run: `npm run bench -- --seed 5 --per 8`.

## Pricing

| | Individual | Team |
|---|---|---|
| Price | **$0 forever** | **$10 / seat / month** |
| Extension, MCP, Skill, CLI | ✓ | ✓ |
| Public + your own private repos | ✓ (your token) | ✓ |
| AI answers | Bring your own Anthropic key | Hosted, no keys to hand out |
| Team keys for CI agents & MCP `ask` | — | ✓ |
| Shared limits, key rotation, one invoice | — | ✓ |

[Request Team access →](https://github.com/in-c0/gitsum/issues/new?template=team.yml&title=Team+plan+request)

## Development

```sh
npm install
npm test                 # engine, MCP round-trip via the official SDK client, Team API (in-memory fakes)
npm run bench            # answer-quality judge on 10 real repos
npm run build:extension  # → extension/dist + extension/gitsum-extension.zip
npm run judge:hud        # Playwright: one-screen layout, live counters, zero console errors
```

| Path | What |
|---|---|
| `src/core/` | The engine every surface shares |
| `src/cli.js`, `src/mcp.js` | CLI and MCP server |
| `skills/gitsum/`, `.claude-plugin/`, `.mcp.json` | Agent Skill and Claude Code plugin |
| `extension/` | Chrome MV3 side panel |
| `api/team/`, `server/`, `supabase/` | Team plan (Vercel functions, Supabase, Stripe, Upstash) |
| `site/` | Website (GitHub Pages) |

The Team backend shares Supabase, Stripe and Upstash with the Gitsum web app; see [`.env.example`](.env.example).

## License 📝

MIT — see [LICENSE](LICENSE).
