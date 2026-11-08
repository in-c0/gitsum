// Service worker: opens the side panel, and tells it which repository the active tab shows.
import { parseRepoInput } from "../../src/core/github.js";

chrome.runtime.onInstalled.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
});
chrome.runtime.onStartup?.addListener(() => {
  chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true }).catch(() => {});
});

/** { owner, repo } for github.com repository pages, else null (settings, explore, gists, other sites). */
function repoOf(url) {
  if (!url || !/^https:\/\/github\.com\/[^/]+\/[^/?#]+/.test(url)) return null;
  return parseRepoInput(url);
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab || null;
}

function announce(tab) {
  if (!tab) return;
  const repo = repoOf(tab.url);
  chrome.runtime
    .sendMessage({ type: "gitsum:active", tabId: tab.id, windowId: tab.windowId, url: repo ? tab.url : "", repo })
    .catch(() => {}); // no panel open: nothing to tell
}

chrome.tabs.onActivated.addListener(async ({ tabId }) => {
  announce(await chrome.tabs.get(tabId).catch(() => null));
});
chrome.tabs.onUpdated.addListener(async (tabId, info, tab) => {
  if (!info.url && info.status !== "complete") return;
  const active = await activeTab();
  if (active?.id === tabId) announce(tab);
});
chrome.windows?.onFocusChanged.addListener(async () => announce(await activeTab()));

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  if (msg?.type === "gitsum:open" && sender.tab?.id != null) {
    // must run synchronously inside the user gesture forwarded from the page button
    chrome.sidePanel.open({ tabId: sender.tab.id }).catch(() => chrome.sidePanel.open({ windowId: sender.tab.windowId }).catch(() => {}));
    announce(sender.tab);
    reply({ ok: true });
    return false;
  }
  if (msg?.type === "gitsum:active?") {
    activeTab().then((tab) => {
      const repo = tab ? repoOf(tab.url) : null;
      reply(tab ? { tabId: tab.id, windowId: tab.windowId, url: repo ? tab.url : "", repo } : {});
    });
    return true; // async reply
  }
  return false;
});
