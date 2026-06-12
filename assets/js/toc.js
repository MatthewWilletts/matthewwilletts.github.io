(function () {
  function init() {
    const article = document.querySelector("article.post");
    const toc = document.getElementById("toc");
    if (!article || !toc) return;

    const heads = Array.from(
      article.querySelectorAll("h1:not(.post-title), h2, h3")
    ).filter(function (h) { return h.textContent.trim(); });

    if (heads.length < 3) { toc.remove(); return; }

    const minLevel = Math.min.apply(null, heads.map(function (h) {
      return +h.tagName[1];
    }));

    const list = document.createElement("ul");
    const links = new Map();
    heads.forEach(function (h, i) {
      if (!h.id) h.id = "toc-" + i;
      const li = document.createElement("li");
      li.className = "toc-l" + (+h.tagName[1] - minLevel);
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      li.appendChild(a);
      list.appendChild(li);
      links.set(h, a);
    });
    toc.appendChild(list);

    let current = null;
    function onScroll() {
      const fromTop = window.scrollY + 96;
      let active = heads[0];
      for (const h of heads) {
        if (h.offsetTop <= fromTop) active = h; else break;
      }
      const link = links.get(active);
      if (link === current) return;
      if (current) current.classList.remove("active");
      current = link;
      if (current) current.classList.add("active");
    }
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
