// Wikipedia-style hover popovers for inline citations.
// Hovering an inline "[n]" cite (a link inside .cite pointing at #ref-n) shows a small card with
// that reference's text + DOI. The card content IS the rendered reference-list entry, so it stays
// in sync automatically. Pure vanilla, no deps. Web-only (the PDF has no scripts).
(function () {
  var pop = document.createElement("div");
  pop.className = "cite-popover";
  var timer = null;
  var attached = false;

  function cancel() { if (timer) { clearTimeout(timer); timer = null; } }
  function hideSoon() { cancel(); timer = setTimeout(function () { pop.style.display = "none"; }, 200); }

  function show(a) {
    if (!attached) { document.body.appendChild(pop); attached = true; }
    cancel();
    var li = document.getElementById((a.getAttribute("href") || "").slice(1));
    if (!li) return;
    pop.innerHTML = li.innerHTML;          // reuse the reference entry: author, year, title, DOI link
    pop.style.display = "block";
    var r = a.getBoundingClientRect();
    var vw = document.documentElement.clientWidth;
    var left = Math.max(8, Math.min(r.left, vw - pop.offsetWidth - 12));
    pop.style.left = (left + window.scrollX) + "px";
    pop.style.top = (r.bottom + window.scrollY + 6) + "px";
  }

  // Delegated so it works regardless of load order and for any number of cites.
  document.addEventListener("mouseover", function (e) {
    var a = e.target.closest ? e.target.closest('.cite a[href^="#ref-"]') : null;
    if (a) { show(a); return; }
    if (pop.contains(e.target)) cancel();  // hovering the card keeps it open, so the DOI is clickable
  });
  document.addEventListener("mouseout", function (e) {
    var a = e.target.closest ? e.target.closest('.cite a[href^="#ref-"]') : null;
    if (a || pop.contains(e.target)) hideSoon();
  });
})();
