// chrome.storage.local in the extension; localStorage when the panel is opened as a plain page (fixtures, judge).
const hasChrome = typeof chrome !== "undefined" && chrome.storage?.local;

export async function get(key, fallback) {
  try {
    if (hasChrome) {
      const v = await chrome.storage.local.get(key);
      return v[key] === undefined ? fallback : v[key];
    }
    const raw = localStorage.getItem(`gitsum:${key}`);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export async function set(key, value) {
  try {
    if (hasChrome) await chrome.storage.local.set({ [key]: value });
    else localStorage.setItem(`gitsum:${key}`, JSON.stringify(value));
  } catch {
    /* storage unavailable: keep working without persistence */
  }
}
