# Gitsum everywhere — SPEC

## 0. Ambition (be-ambitious)

**Goal beyond the goal:** any person *or agent* understands an unfamiliar GitHub repo in under 10 seconds, without leaving where they already are (GitHub tab, Claude Code, Cursor, any agent). Teams pay because onboarding hours and agent token bills drop.

**Types used:** Altitude (it serves an outcome: adoption + team revenue), Depth (answers must be *right*, judged by a benchmark, not by eye), Range (distribution borrowed from the five fastest-growing repo/agent tools). Speed skipped.

**Altitude ladder**
1. One engine, three front doors: Chrome extension (side panel HUD), MCP server, Agent Skill — all free for individuals.
2. Agents pick Gitsum over crawling files themselves because it is measurably cheaper: the right `path:line` in fewer tokens than "read README + list tree".
3. Teams share one paid workspace: hosted AI answers with no keys, team keys for CI agents, one invoice per seat.

**Depth — what a top practitioner would do**
- Start from the raw tree + raw files, not GitHub's HTML.
- Use IR instruments: BM25 over chunked key files, a definition detector (`function|def|class|fn… <ident>`), path-name match, and GitHub code search as a fallback when a token exists.
- Outside judge: `bench/run.mjs` generates questions from *real* repos ("Where is `X` defined?" for identifiers sampled from source files the engine never saw as a hint) and scores hit@3 against the true file. Also measures tokens returned vs the naive baseline.
- **Hard target:** hit@3 ≥ 80 % on definition questions across 10 popular repos; overview ≤ 2 s warm; every MCP tool response ≤ 4 k tokens.

**Range — verbs, not nouns:** "make a stranger's codebase legible fast, inside the tool you're already in".
| Field | Job they'd actually do | Method/tool |
|---|---|---|
| Cartographer | Show the whole shape before any detail | Exploded strata of the real tree; treemap lens (d3-hierarchy) |
| Librarian / IR engineer | Return the shelf mark, not the book | BM25 + definition index + code-search fallback, `path:line` citations |
| DevRel for agents | Remove every install step | `npx -y github:in-c0/gitsum` (no npm publish needed), one-line Skill URL prompt (MoneyPrinterTurbo pattern), `claude mcp add` one-liner |

## 1. Five references (fast-growing, same job) → common patterns
| Reference | Pattern borrowed | Where in Gitsum |
|---|---|---|
| DeepWiki (Cognition) | URL swap `github.com → deepwiki.com`; MCP of the same data | Gitsum site already supports `/<owner>/<repo>`; MCP exposes the same overview |
| gitingest | `hub → ingest` swap + Chrome extension + CLI, all free, no sign-up | Extension button on every repo page; CLI `gitsum overview o/r` |
| Context7 (Upstash) | One-line `npx` MCP install, free for individuals, paid for teams/higher limits | `claude mcp add gitsum -- npx -y github:in-c0/gitsum mcp`; Team plan |
| repomix | Same engine shipped as CLI, MCP, extension, website | Single `src/core` used by every surface |
| MoneyPrinterTurbo | "Use this Skill: <raw SKILL.md URL> + task" one-liner for any agent; README shows every front door with a screenshot | `skills/gitsum/SKILL.md`; README structure |

**Common pattern:** core value free with zero sign-up; meet users inside an existing surface (GitHub page, agent); one-line install; pay for team/scale, never for the first "aha".

## 2. Pricing
- **Individual — free forever:** extension, MCP, Skill, CLI. Public repos, plus your own private repos with your own `GITHUB_TOKEN`. AI answers in the extension with your own Anthropic key (stored only in `chrome.storage.local`). Agents need no Gitsum AI: the MCP returns evidence and the calling model answers.
- **Team — paid per seat (default $10/seat/month, env `GITSUM_TEAM_SEAT_PRICE_ID`):** hosted AI answers for every seat with no keys, team keys (`gsk_…`) for CI agents and the MCP `ask` tool, shared usage limits, one Stripe invoice, admin can rotate keys. Server: `api/team/*`.
- License: MIT for the clients (distribution > lock-in); revenue comes from the hosted Team service.

## 3. Extension HUD (living-data-hud)
**Identity:** looks like a geological core sample because a repo *is* layers of decisions — top-level folders are strata, files are grains sized by bytes and tinted by GitHub's own linguist language colour. Monospace paths, slate/paper palette, one soft tick sound on lens switch.

**Acts**
1. *Shape* — open on any repo page → strata assemble as the tree arrives; counter "N files · M folders" counts up from real data. One job: show the shape.
2. *Ask* — one input "Ask this repo". Matching grains light while typing (< 300 ms, path + token match); Enter → answer with `path:line` chips; a lineage thread links each chip to its grain.
3. *Read* — the edge readouts assemble: stack chips, 6 health dots, run command (copy), languages bar. Selection line: "lit: 3 files · ‘auth’ · ×".

**Lenses:** Strata ↔ Atlas (squarified treemap) on the same data, switchable without leaving the screen.

**Data:** GitHub REST (tree + meta) and raw files. Numbers are *counted* (from the tree) or *live* (re-fetched, stamped). No invented figures.

**Predictable failures handled:** non-repo GitHub page (prompt to open a repo), 404/private (offer token field), rate limit (minutes to reset + token hint), truncated tree (labelled "partial"), 1–2 letter query ("keep typing"), no match (closest path by edit distance, Tab to accept), huge monorepo (cap grains, label "top N by size"), theme never auto-toggles.

## 4. Judge checks
- `npm test` — core unit tests, MCP stdio round-trip via the official SDK client, team API with fakes.
- `bench/run.mjs` — hit@3 on 10 real repos, token budget per response.
- `scripts/judge-hud.mjs` — Playwright (Chrome) at 1440×900, 1280×720, 390×844 and the 400×800 side panel: no document scroll, nothing outside the viewport, counters equal a recount of the fixture tree, lit grains update < 300 ms after typing, zero console errors, lens switch works, theme stable.
