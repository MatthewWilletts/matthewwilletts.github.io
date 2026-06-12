(function () {
  const storageKey = "theme";
  const settings = ["system", "light", "dark"];
  const themeColors = {
    light: "#faf8f1",
    dark: "#121416",
  };

  function readStoredTheme() {
    try {
      return localStorage.getItem(storageKey);
    } catch (_error) {
      return null;
    }
  }

  function writeStoredTheme(setting) {
    try {
      localStorage.setItem(storageKey, setting);
    } catch (_error) {
      /* Theme still works for this page load. */
    }
  }

  function getSetting() {
    const stored = readStoredTheme();
    return settings.includes(stored) ? stored : "system";
  }

  function getComputedTheme(setting) {
    if (setting === "system") {
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    return setting;
  }

  function applyTheme(setting, transition) {
    const computed = getComputedTheme(setting);
    const root = document.documentElement;
    root.setAttribute("data-theme-setting", setting);
    root.setAttribute("data-theme", computed);

    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) themeColor.setAttribute("content", themeColors[computed]);

    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
      toggle.setAttribute("aria-label", "Theme: " + setting + ". Click to change.");
      toggle.setAttribute("title", "Theme: " + setting);
    }

    if (transition) {
      root.classList.add("theme-transition");
      window.setTimeout(function () {
        root.classList.remove("theme-transition");
      }, 250);
    }
  }

  function setSetting(setting, transition) {
    writeStoredTheme(setting);
    applyTheme(setting, transition);
  }

  function toggleTheme() {
    const current = getSetting();
    const next = settings[(settings.indexOf(current) + 1) % settings.length];
    setSetting(next, true);
  }

  setSetting(getSetting(), false);

  document.addEventListener("DOMContentLoaded", function () {
    const toggle = document.getElementById("theme-toggle");
    if (toggle) toggle.addEventListener("click", toggleTheme);
    applyTheme(getSetting(), false);
  });

  if (window.matchMedia) {
    const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
    const onSystemThemeChange = function () {
      if (getSetting() === "system") applyTheme("system", true);
    };
    if (systemTheme.addEventListener) {
      systemTheme.addEventListener("change", onSystemThemeChange);
    } else if (systemTheme.addListener) {
      systemTheme.addListener(onSystemThemeChange);
    }
  }
})();
