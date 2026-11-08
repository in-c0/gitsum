// Loads the unpacked extension (extension/dist) in real Chrome and checks that the service worker
// registers, the side panel page boots inside the extension origin, and nothing logs an error.
//   node scripts/check-extension-load.mjs   (run `npm run build:extension` first)
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

const dist = join(dirname(fileURLToPath(import.meta.url)), "..", "extension", "dist");
const ctx = await chromium.launchPersistentContext(mkdtempSync(join(tmpdir(), "gitsum-ext-")), {
  executablePath: process.env.CHROME || "/usr/bin/google-chrome",
  headless: true,
  args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`],
});
const errors = [];
let [sw] = ctx.serviceWorkers();
if (!sw) sw = await ctx.waitForEvent("serviceworker", { timeout: 15000 });
const id = new URL(sw.url()).host;
console.log(`service worker: ${sw.url()}`);
const swInfo = await sw.evaluate(async () => ({
  manifest: chrome.runtime.getManifest().name,
  behavior: await chrome.sidePanel.getPanelBehavior(),
  sidePanelOptions: await chrome.sidePanel.getOptions({}),
}));
console.log("manifest name:", swInfo.manifest, "· panel behavior:", JSON.stringify(swInfo.behavior), "· panel:", JSON.stringify(swInfo.sidePanelOptions));

const page = await ctx.newPage();
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(`chrome-extension://${id}/sidepanel.html`);
await page.waitForSelector("#notice:not([hidden])", { timeout: 10000 }).catch(async (e) => {
  console.log("DEBUG", await page.evaluate(() => ({ state: document.body.dataset.state, act: document.body.dataset.act, notice: document.getElementById("notice").hidden })), errors);
  throw e;
});
const title = await page.textContent("#noticeTitle");
console.log(`side panel boots in extension origin: notice = "${title}"`);
const active = await page.evaluate(() => chrome.runtime.sendMessage({ type: "gitsum:active?" }));
console.log("background answers gitsum:active? →", JSON.stringify(active));
const stored = await page.evaluate(async () => {
  await chrome.storage.local.set({ __probe: 1 });
  return (await chrome.storage.local.get("__probe")).__probe;
});
console.log("chrome.storage.local round-trip:", stored === 1 ? "ok" : "FAILED");

// Content script on a github.com repo page. github.com itself may be unreachable from CI sandboxes, so the
// request is answered with a minimal page carrying GitHub's real repo-header markup (the selectors we rely on).
await ctx.route("https://github.com/**", (route) =>
  route.fulfill({
    contentType: "text/html",
    body: `<!doctype html><meta name="octolytics-dimension-repository_nwo" content="in-c0/gitsum"><div id="repository-container-header"><ul class="pagehead-actions"><li><button class="btn btn-sm">Star</button></li></ul></div><a id="spa" href="/in-c0/gitsum/blob/main/README.md">readme</a>`,
  })
);
const gh = await ctx.newPage();
gh.on("pageerror", (e) => errors.push(String(e)));
await gh.goto("https://github.com/in-c0/gitsum");
await gh.waitForSelector("#gitsum-open-panel button", { timeout: 10000 });
// SPA navigation: GitHub swaps the header; the button must come back exactly once
await gh.evaluate(() => {
  document.getElementById("repository-container-header").outerHTML = '<div id="repository-container-header"><ul class="pagehead-actions"><li><button class="btn btn-sm">Star</button></li></ul></div>';
  history.pushState({}, "", "/in-c0/gitsum/tree/main/src");
  document.dispatchEvent(new Event("turbo:load"));
});
await gh.waitForSelector("#gitsum-open-panel button", { timeout: 5000 });
const buttons = await gh.locator("#gitsum-open-panel").count();
console.log(`content script button after SPA swap: ${buttons} (want 1)`);
await gh.bringToFront();
const seen = await page.evaluate(() => chrome.runtime.sendMessage({ type: "gitsum:active?" }));
console.log("background sees active repo tab →", JSON.stringify(seen.repo), seen.url);
// clicking the page button asks the background to open the panel (no error expected; headless has no visible panel)
await gh.click("#gitsum-open-panel button");
await gh.waitForTimeout(300);
// the open panel follows the active tab: it should now be loading/showing in-c0/gitsum (live GitHub API)
await page.waitForFunction(() => document.getElementById("repoName").textContent === "gitsum", null, { timeout: 8000 }).catch(() => {});
await page.waitForFunction(() => ["ready", "error"].includes(document.body.dataset.state), null, { timeout: 15000 }).catch(() => {});
const followed = await page.evaluate(() => ({ repo: repoOwner.textContent + repoName.textContent, state: document.body.dataset.state, files: nFiles.textContent, notice: notice.hidden ? "" : noticeTitle.textContent + " " + noticeHint.textContent }));
console.log("panel followed the tab →", JSON.stringify(followed));
await ctx.close();
// HTTP failures from the live GitHub API (sandbox rate limits) are environment, not code: reported, not failed
const network = errors.filter((e) => /Failed to load resource/.test(e));
if (network.length) console.log(`live network responses (not code errors): ${network.join(" | ")}`);
errors.splice(0, errors.length, ...errors.filter((e) => !/Failed to load resource/.test(e)));
const ok =
  swInfo.behavior?.openPanelOnActionClick === true &&
  title === "Open a GitHub repository" &&
  stored === 1 &&
  buttons === 1 &&
  seen.repo?.owner === "in-c0" &&
  seen.repo?.repo === "gitsum" &&
  followed.repo === "in-c0/gitsum" &&
  (followed.state === "ready" || !!followed.notice) &&
  !errors.length;
console.log(ok ? "EXTENSION LOAD: PASS" : `EXTENSION LOAD: FAIL ${errors.join(" | ")}`);
process.exit(ok ? 0 : 1);
