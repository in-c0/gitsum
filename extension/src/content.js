// One small "Gitsum" button next to the repo header actions (Watch / Fork / Star). Opens the side panel.
// Idempotent and resilient to GitHub's SPA navigation (turbo:load, popstate, DOM mutations).
(() => {
  const ID = "gitsum-open-panel";
  const RESERVED = /^(settings|orgs|marketplace|explore|topics|trending|collections|sponsors|notifications|pulls|issues|codespaces|features|about|pricing|login|signup|new|organizations|enterprise|apps|search|users|site|security|readme|customer-stories|team)$/i;

  const isRepoPage = () => {
    const [owner, repo] = location.pathname.split("/").filter(Boolean);
    return !!(owner && repo && !RESERVED.test(owner) && document.querySelector('meta[name="octolytics-dimension-repository_nwo"], meta[name="go-import"], #repository-container-header, [data-testid="repository-header"]'));
  };

  const anchor = () =>
    document.querySelector("#repository-container-header ul.pagehead-actions") ||
    document.querySelector("ul.pagehead-actions") ||
    document.querySelector('[data-testid="repository-header-actions"], [data-testid="repo-header-actions"]');

  function button(list) {
    const li = document.createElement(list.tagName === "UL" ? "li" : "div");
    li.id = ID;
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn btn-sm";
    b.title = "Understand this repository in the Gitsum side panel";
    b.setAttribute("aria-label", "Open Gitsum");
    b.style.cssText = "display:inline-flex;align-items:center;gap:6px";
    b.innerHTML =
      '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" style="fill:currentColor;opacity:.85"><path d="M3 3.5h9l1.5 1.6H4.5z"/><path d="M2.5 7.2h9.5l1.5 1.6H4z"/><path d="M2 10.9h10l1.5 1.6H3.5z"/></svg><span>Gitsum</span>';
    b.addEventListener("click", () => {
      try {
        chrome.runtime.sendMessage({ type: "gitsum:open" });
      } catch {
        /* extension reloaded: the page needs a refresh */
      }
    });
    li.append(b);
    return li;
  }

  function ensure() {
    const existing = document.getElementById(ID);
    if (!isRepoPage()) {
      existing?.remove();
      return;
    }
    const list = anchor();
    if (!list) return;
    if (existing && list.contains(existing)) return;
    existing?.remove();
    list.prepend(button(list));
  }

  let queued = false;
  const schedule = () => {
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      ensure();
    });
  };
  ensure();
  document.addEventListener("turbo:load", schedule);
  document.addEventListener("turbo:render", schedule);
  window.addEventListener("popstate", schedule);
  new MutationObserver(schedule).observe(document.documentElement, { childList: true, subtree: true });
})();
