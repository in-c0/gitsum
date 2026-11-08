// Outside judge for the extension's Living-Data HUD (SPEC §4). Headless Chrome via Playwright.
//   npm run judge:hud            (builds extension/dist with fixtures first)
// Env: CHROME=/path/to/chrome (default /usr/bin/google-chrome), SHOTS=dir for screenshots, ONLY=viewport-name.
import { execFileSync } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { basename, dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(root, "extension", "dist");
const SHOTS = process.env.SHOTS || join(tmpdir(), "gitsum-hud-shots");
const CHROME = process.env.CHROME || "/usr/bin/google-chrome";
mkdirSync(SHOTS, { recursive: true });

execFileSync(process.execPath, [join(root, "extension", "build.mjs"), "--fixtures"], { stdio: "inherit" });

// ---------- independent recount helpers (do not import the panel's code) ----------
const fxSmall = JSON.parse(readFileSync(join(dist, "fixtures", "repo.json"), "utf8"));
const fxLarge = JSON.parse(readFileSync(join(dist, "fixtures", "large.json"), "utf8"));
function recount(files) {
  const dirs = new Set();
  for (const { path } of files) {
    const parts = path.split("/");
    for (let i = 1; i < parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
  }
  return { files: files.length, folders: dirs.size };
}
const recountMatch = (files, term) => files.filter((f) => f.path.toLowerCase().includes(term)).length;

// ---------- static server: dist + the large fixture's clone for raw file reads ----------
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml" };
// file contents come from local clones of the fixture repos (recorded in each fixture as cloneDir)
// HUD_CLONES=<dir> (CI) looks for each clone as <dir>/<basename>; scripts/fetch-hud-clones.mjs creates them at the recorded sha.
const cloneDirOf = (f) => (process.env.HUD_CLONES ? join(process.env.HUD_CLONES, basename(f.cloneDir)) : f.cloneDir);
const clones = [fxSmall, fxLarge].filter((f) => f.rawBase && f.cloneDir).map((f) => [`/${f.rawBase}`, cloneDirOf(f)]);
for (const [, dir] of clones) if (!existsSync(dir)) throw new Error(`Fixture clone missing: ${dir} — run: node scripts/fetch-hud-clones.mjs`);
const safe = (p) => normalize(p).replace(/^(\.\.[/\\])+/, "");
const server = createServer((req, res) => {
  const url = new URL(req.url, "http://x");
  const p = decodeURIComponent(url.pathname);
  const clone = clones.find(([prefix]) => p.startsWith(prefix));
  const file = clone ? join(clone[1], safe(p.slice(clone[0].length))) : join(dist, safe(p === "/" ? "/sidepanel.html" : p));
  if (!existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "content-type": TYPES[extname(file)] || "text/plain; charset=utf-8" });
  createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const base = `http://127.0.0.1:${server.address().port}`;

const VIEWPORTS = [
  { name: "panel-360x700", width: 360, height: 700 },
  { name: "panel-400x800", width: 400, height: 800 },
  { name: "panel-500x900", width: 500, height: 900 },
  { name: "desktop-1280x720", width: 1280, height: 720 },
  { name: "desktop-1440x900", width: 1440, height: 900 },
  { name: "phone-390x844", width: 390, height: 844 },
].filter((v) => !process.env.ONLY || v.name.includes(process.env.ONLY));

const results = [];
const check = (vp, name, ok, detail = "") => {
  results.push({ vp, name, ok: !!ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${vp.padEnd(18)} ${name}${detail ? `  — ${detail}` : ""}`);
};

const browser = await chromium.launch({ executablePath: CHROME, headless: true });

async function fresh(vp, query, { colorScheme = "dark" } = {}) {
  const ctx = await browser.newContext({ viewport: { width: vp.width, height: vp.height }, colorScheme, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  // block anything that isn't the local server: the fixture mode must be fully offline
  await ctx.route(/^(?!http:\/\/127\.0\.0\.1)/, (route) => route.abort());
  await page.goto(`${base}/sidepanel.html?${query}`);
  return { ctx, page, errors };
}

async function layoutProblems(page) {
  return page.evaluate(() => {
    const out = [];
    const de = document.documentElement;
    if (de.scrollHeight > innerHeight + 1) out.push(`vertical scroll ${de.scrollHeight}>${innerHeight}`);
    if (de.scrollWidth > innerWidth + 1) out.push(`horizontal scroll ${de.scrollWidth}>${innerWidth}`);
    const ids = ["top", "count", "stage", "field", "selection", "answer", "readouts", "roLang", "roHealth", "roStack", "roRun", "roLive", "lens", "gear"];
    for (const id of ids) {
      const n = document.getElementById(id);
      if (!n || n.hidden || n.closest("[hidden]")) continue;
      const r = n.getBoundingClientRect();
      if (!r.width || !r.height) continue;
      if (r.left < -0.5 || r.top < -0.5 || r.right > innerWidth + 0.5 || r.bottom > innerHeight + 0.5) out.push(`#${id} outside viewport (${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.right)},${Math.round(r.bottom)})`);
    }
    // primary panels must not overlap each other
    // invisible (opacity 0, not yet revealed) panels can't collide visually and take no pointer events
    const prim = ["count", "stage", "ask", "answer", "readouts"].map((id) => [id, document.getElementById(id)]).filter(([, n]) => n && !n.hidden && getComputedStyle(n).opacity !== "0" && getComputedStyle(n).pointerEvents !== "none");
    for (let i = 0; i < prim.length; i++)
      for (let j = i + 1; j < prim.length; j++) {
        const a = prim[i][1].getBoundingClientRect();
        const b = prim[j][1].getBoundingClientRect();
        const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left);
        const oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
        if (ox > 1 && oy > 1) out.push(`#${prim[i][0]} overlaps #${prim[j][0]} by ${Math.round(oy)}px`);
      }
    const stage = document.getElementById("stage").getBoundingClientRect();
    if (stage.height < 160) out.push(`hero stage too small (${Math.round(stage.height)}px)`);
    return out;
  });
}

const shot = (page, vp, act) => page.screenshot({ path: join(SHOTS, `${vp.name}-${act}.png`) });

for (const vp of VIEWPORTS) {
  // ---------- small real fixture: all three acts ----------
  {
    const { ctx, page, errors } = await fresh(vp, "fixture=1");
    const expect = recount(fxSmall.repoData.files);
    await page.waitForFunction(() => document.getElementById("counter").dataset.final === "1", null, { timeout: 8000 });
    await shot(page, vp, "1-shape");
    const shown = await page.evaluate(() => ({ files: Number(nFiles.textContent.replace(/,/g, "")), folders: Number(nFolders.textContent.replace(/,/g, "")), drawn: window.__gitsum.hero.drawCount }));
    check(vp.name, "counters equal recount", shown.files === expect.files && shown.folders === expect.folders, `shown ${shown.files}/${shown.folders}, recount ${expect.files}/${expect.folders}`);
    check(vp.name, "every file drawn as a grain", shown.drawn === expect.files, `${shown.drawn} grains`);

    await page.waitForFunction(() => document.body.dataset.act === "3" && document.body.dataset.toured === "1", null, { timeout: 12000 });
    await page.waitForTimeout(700);
    let lp = await layoutProblems(page);
    check(vp.name, "act 3 layout fits (no scroll, in viewport, no overlap)", !lp.length, lp.join("; "));
    const themeBefore = await page.evaluate(() => document.documentElement.dataset.theme + "|" + getComputedStyle(document.body).backgroundColor);
    await shot(page, vp, "3-read");

    // Act 2: typing lights grains
    const term = "api";
    const want = recountMatch(fxSmall.repoData.files, term);
    await page.click("#q");
    const t0 = Date.now();
    await page.keyboard.type(term, { delay: 0 });
    await page.waitForFunction((t) => document.getElementById("selection").dataset.q === t, term, { timeout: 2000 });
    const lit = await page.evaluate(() => ({ n: Number(selection.dataset.lit), ms: Number(selection.dataset.ms), text: selection.textContent, heroLit: window.__gitsum.hero.lit?.size ?? -1 }));
    const wall = Date.now() - t0;
    check(vp.name, `typing "${term}" lights matching grains`, lit.n === want && lit.heroLit === want, `lit ${lit.n} (hero ${lit.heroLit}), recount ${want}; "${lit.text}"`);
    check(vp.name, "lit within 300 ms of last keystroke", lit.ms < 300 && wall < 1500, `${lit.ms} ms after input (wall ${wall} ms incl. typing)`);
    await shot(page, vp, "2-typing");

    // short + no-match + Tab
    await page.fill("#q", "ap");
    await page.waitForFunction(() => selection.dataset.state === "short");
    check(vp.name, "1–2 letters → keep typing", (await page.textContent("#selection")).includes("keep typing"));
    await page.fill("#q", "chekout");
    await page.waitForFunction(() => selection.dataset.state === "none");
    const sug = await page.textContent("#selection");
    await page.focus("#q");
    await page.keyboard.press("Tab");
    await page.waitForFunction(() => selection.dataset.state === "ok", null, { timeout: 2000 }).catch(() => {});
    const afterTab = await page.inputValue("#q");
    check(vp.name, "no match → closest path, Tab accepts", /closest/.test(sug) && /checkout/i.test(afterTab), `"${sug}" → "${afterTab}"`);

    // Enter → answer with chips + lineage
    await page.fill("#q", "Where is the Stripe webhook handled?");
    await page.keyboard.press("Enter");
    await page.waitForSelector("#chips .chip", { timeout: 8000 });
    await page.waitForTimeout(250);
    const ans = await page.evaluate(() => ({ chips: [...document.querySelectorAll("#chips .chip")].map((c) => c.textContent), threads: document.querySelectorAll("#lineage path").length, kind: answerKind.textContent }));
    check(vp.name, "Enter → answer with path:line chips + lineage threads", ans.chips.length > 0 && ans.threads > 0, `${ans.chips.slice(0, 3).join(", ")} · ${ans.threads} threads · ${ans.kind}`);
    lp = await layoutProblems(page);
    check(vp.name, "answer layout fits", !lp.length, lp.join("; "));
    await shot(page, vp, "2-answer");
    await page.click("#chips .chip");
    const opened = await page.evaluate(() => window.__gitsumOpened || "");
    check(vp.name, "chip click opens the GitHub URL", /^https:\/\/github\.com\/.+\/blob\//.test(opened), opened);

    // lens switch changes drawn layout
    await page.click("#answerClose");
    const sigA = await page.evaluate(() => window.__gitsum.hero.signature());
    const lensNow = await page.evaluate(() => window.__gitsum.hero.lens);
    await page.click(`#lens [data-lens="${lensNow === "strata" ? "atlas" : "strata"}"]`);
    await page.waitForTimeout(700);
    const sigB = await page.evaluate(() => window.__gitsum.hero.signature());
    check(vp.name, "lens switch changes the drawn layout", sigA !== sigB && sigA.split(":")[1] !== sigB.split(":")[1], `${sigA} → ${sigB}`);
    await page.fill("#q", "");
    await page.keyboard.press("Escape");
    await page.waitForTimeout(150);
    await shot(page, vp, "3-atlas");
    await page.click(`#lens [data-lens="${lensNow}"]`);

    // recent questions survive reload and delete with ×
    await page.reload();
    await page.waitForFunction(() => document.querySelectorAll("#selection .recent:not(.suggest)").length > 0, null, { timeout: 8000 }).catch(() => {});
    const recents = await page.$$eval("#selection .recent:not(.suggest) > span", (n) => n.map((x) => x.textContent));
    check(vp.name, "recent questions survive reload", recents.includes("Where is the Stripe webhook handled?"), recents.join(" | "));
    await page.hover("#selection .recent:not(.suggest)");
    await page.click("#selection .recent:not(.suggest) .x");
    await page.waitForTimeout(100);
    await page.reload();
    await page.waitForFunction(() => document.getElementById("counter").dataset.final === "1", null, { timeout: 8000 });
    await page.waitForTimeout(400);
    const after = await page.$$eval("#selection .recent:not(.suggest) > span", (n) => n.map((x) => x.textContent));
    check(vp.name, "× deletes a recent question (persisted)", !after.includes("Where is the Stripe webhook handled?"), after.join(" | ") || "(none)");

    // theme never changes by itself
    if (vp === VIEWPORTS[0] || process.env.LONG) {
      await page.waitForTimeout(10_000);
    } else await page.waitForTimeout(1500);
    const themeAfter = await page.evaluate(() => document.documentElement.dataset.theme + "|" + getComputedStyle(document.body).backgroundColor);
    check(vp.name, `theme stable while idle (${vp === VIEWPORTS[0] || process.env.LONG ? "10 s" : "1.5 s"})`, themeBefore === themeAfter, themeAfter);
    check(vp.name, "zero console errors (small fixture)", !errors.length, errors.slice(0, 3).join(" | "));
    await ctx.close();
  }

  // ---------- large fixture: caps + performance ----------
  {
    const { ctx, page, errors } = await fresh(vp, "fixture=large", { colorScheme: "light" });
    const expect = recount(fxLarge.repoData.files);
    await page.waitForFunction(() => document.getElementById("counter").dataset.final === "1", null, { timeout: 15000 });
    const shown = await page.evaluate(() => ({ files: Number(nFiles.textContent.replace(/,/g, "")), folders: Number(nFolders.textContent.replace(/,/g, "")), drawn: window.__gitsum.hero.drawCount, badges: badges.textContent }));
    check(vp.name, "large: counters equal recount", shown.files === expect.files && shown.folders === expect.folders, `shown ${shown.files}/${shown.folders}, recount ${expect.files}/${expect.folders}`);
    check(vp.name, "large: grains capped + labelled", shown.drawn === 5000 && /top 5,000 by size/.test(shown.badges), `${shown.drawn} grains · "${shown.badges}"`);
    await page.waitForFunction(() => document.body.dataset.act === "3", null, { timeout: 15000 });
    await page.waitForTimeout(1200);
    const term = "route";
    const want = recountMatch(fxLarge.repoData.files, term);
    await page.click("#q");
    await page.keyboard.type(term, { delay: 0 });
    await page.waitForFunction((t) => document.getElementById("selection").dataset.q === t, term, { timeout: 3000 });
    const lit = await page.evaluate(() => ({ n: Number(selection.dataset.lit), ms: Number(selection.dataset.ms), text: selection.textContent }));
    check(vp.name, `large: typing "${term}" lights ${want} paths within 300 ms`, lit.n === want && lit.ms < 300, `lit ${lit.n}, recount ${want}, ${lit.ms} ms; "${lit.text}"`);
    // cost of one full redraw of 5k grains (mid lens-morph, the most expensive frame)
    const fps = await page.evaluate(async () => {
      const h = window.__gitsum.hero;
      const times = [];
      for (let n = 0; n < 60; n++) {
        h.setLens(n % 20 < 10 ? "atlas" : "strata");
        const t = performance.now();
        h.draw();
        times.push(performance.now() - t);
        await new Promise((r) => requestAnimationFrame(r));
      }
      times.sort((a, b) => a - b);
      h.setLens(document.querySelector('#lens [aria-selected="true"]').dataset.lens); // hand the lens back to the UI
      return { p50: times[30], p90: times[54] };
    });
    check(vp.name, "large: 5k grains redraw < 16 ms (p90, headless software GL)", fps.p90 < 16, `p50 ${fps.p50.toFixed(1)} ms · p90 ${fps.p90.toFixed(1)} ms`);
    await page.click(`#lens [data-lens="strata"]`);
    await page.waitForTimeout(700);
    const lp = await layoutProblems(page);
    check(vp.name, "large: layout fits", !lp.length, lp.join("; "));
    await shot(page, vp, "large-typing");
    await page.fill("#q", "");
    await page.keyboard.press("Escape");
    await page.waitForTimeout(200);
    await shot(page, vp, "large-light");
    // offline ask against the clone
    await page.fill("#q", "Where is `matchRoute` defined?");
    await page.keyboard.press("Enter");
    await page.waitForSelector("#chips .chip", { timeout: 20000 }).catch(() => {});
    const ans = await page.evaluate(() => ({ chips: [...document.querySelectorAll("#chips .chip")].map((c) => c.textContent), text: answerText.textContent }));
    check(vp.name, "large: offline ask() finds a definition", ans.chips.length > 0, `${ans.text.slice(0, 90)} · ${ans.chips[0] || "-"}`);
    check(vp.name, "zero console errors (large fixture)", !errors.length, errors.slice(0, 3).join(" | "));
    await ctx.close();
  }

  // ---------- failure states (once per viewport, quick) ----------
  for (const [q, want, needToken] of [
    ["", "Open a GitHub repository", false],
    ["fixture=404", "Repository not found.", true],
    ["fixture=ratelimit", "GitHub's hourly API limit was reached.", true],
  ]) {
    const { ctx, page, errors } = await fresh(vp, q);
    await page.waitForSelector("#notice:not([hidden])", { timeout: 5000 });
    const n = await page.evaluate(() => ({ title: noticeTitle.textContent, hint: noticeHint.textContent, action: noticeAction.hidden ? "" : noticeAction.textContent }));
    let ok = n.title === want && (!needToken || /token/i.test(n.action));
    if (q === "fixture=ratelimit") ok = ok && /\d+ min/.test(n.hint);
    const lp = await layoutProblems(page);
    check(vp.name, `failure "${q || "non-repo"}" handled`, ok && !lp.length, `${n.title} · ${n.hint} · [${n.action}] ${lp.join("; ")}`);
    if (q === "fixture=404") {
      await page.click("#noticeAction");
      const focused = await page.evaluate(() => !drawer.hidden && document.activeElement?.id);
      check(vp.name, "404 → token field offered", focused === "ghToken", String(focused));
      await page.waitForTimeout(400); // let the drawer finish sliding in
      await shot(page, vp, "settings");
    }
    if (q === "") await shot(page, vp, "0-nonrepo");
    // a refused request in a failure fixture logs a network error by design; anything else is a bug
    const unexpected = errors.filter((e) => !/Failed to load resource/.test(e));
    check(vp.name, `zero console errors (${q || "non-repo"})`, !unexpected.length, unexpected.slice(0, 2).join(" | "));
    await ctx.close();
  }

  // partial tree label
  {
    const { ctx, page } = await fresh(vp, "fixture=1&partial=1");
    await page.waitForFunction(() => document.getElementById("counter").dataset.final === "1", null, { timeout: 8000 });
    check(vp.name, "truncated tree labelled partial", (await page.textContent("#badges")).includes("partial"));
    await ctx.close();
  }
}

await browser.close();
server.close();

const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed · screenshots in ${SHOTS}`);
if (failed.length) {
  console.log("FAILED:");
  for (const f of failed) console.log(`  ${f.vp} ${f.name} — ${f.detail}`);
  process.exit(1);
}
