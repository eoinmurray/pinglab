// Full-viewport gallery for article figures. Unlinked <figure> images are enhanced in place;
// linked images retain their authored destination, and data-lightbox="off" is an explicit opt-out.
(function () {
  var images = Array.prototype.slice.call(document.querySelectorAll("figure img")).filter(
    function (image) {
      return !image.closest("a") && image.getAttribute("data-lightbox") !== "off";
    }
  );
  if (!images.length) return;

  var dialog = document.createElement("dialog");
  dialog.className = "demolab-lightbox";
  dialog.setAttribute("aria-label", "Image viewer");

  var shell = document.createElement("div");
  shell.className = "demolab-lightbox-shell";
  var stage = document.createElement("div");
  stage.className = "demolab-lightbox-stage";
  var fullImage = document.createElement("img");
  fullImage.className = "demolab-lightbox-image";
  fullImage.draggable = false;
  stage.appendChild(fullImage);

  var footer = document.createElement("div");
  footer.className = "demolab-lightbox-footer";
  var caption = document.createElement("p");
  caption.className = "demolab-lightbox-caption";
  var count = document.createElement("p");
  count.className = "demolab-lightbox-count";
  count.setAttribute("aria-live", "polite");
  footer.appendChild(caption);
  footer.appendChild(count);

  function control(className, label, text) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "demolab-lightbox-control " + className;
    button.setAttribute("aria-label", label);
    button.textContent = text;
    return button;
  }
  var closeButton = control("demolab-lightbox-close", "Close image viewer", "\u00d7");
  var previousButton = control("demolab-lightbox-prev", "Previous image", "\u2039");
  var nextButton = control("demolab-lightbox-next", "Next image", "\u203a");
  shell.appendChild(stage);
  shell.appendChild(footer);
  shell.appendChild(closeButton);
  shell.appendChild(previousButton);
  shell.appendChild(nextButton);
  dialog.appendChild(shell);
  if (images.length === 1) dialog.classList.add("demolab-lightbox-single");
  document.body.appendChild(dialog);

  var index = 0;
  var opener = null;
  var touchStart = null;

  function figureCaption(image) {
    var figure = image.closest("figure");
    var node = figure && figure.querySelector("figcaption");
    return node ? node.textContent.replace(/\s+/g, " ").trim() : "";
  }

  function show(nextIndex) {
    index = (nextIndex + images.length) % images.length;
    var source = images[index];
    fullImage.src = source.currentSrc || source.src;
    fullImage.alt = source.alt || "";
    caption.textContent = figureCaption(source) || source.alt || "";
    caption.hidden = !caption.textContent;
    count.textContent = "Image " + (index + 1) + " of " + images.length;
  }

  function open(image) {
    opener = image;
    show(images.indexOf(image));
    document.body.classList.add("demolab-lightbox-open");
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
    window.requestAnimationFrame(function () { closeButton.focus(); });
  }

  function close() {
    if (!dialog.hasAttribute("open")) return;
    if (typeof dialog.close === "function") dialog.close();
    else {
      dialog.removeAttribute("open");
      restorePage();
    }
  }

  function restorePage() {
    document.body.classList.remove("demolab-lightbox-open");
    if (opener) opener.focus();
  }

  images.forEach(function (image) {
    image.classList.add("demolab-lightbox-trigger");
    image.tabIndex = 0;
    image.setAttribute("role", "button");
    image.setAttribute("aria-haspopup", "dialog");
    image.setAttribute("aria-label", "Open image: " + (image.alt || figureCaption(image) || "figure"));
    image.addEventListener("click", function () { open(image); });
    image.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        open(image);
      }
    });
  });

  closeButton.addEventListener("click", close);
  previousButton.addEventListener("click", function () { show(index - 1); });
  nextButton.addEventListener("click", function () { show(index + 1); });
  dialog.addEventListener("close", restorePage);
  dialog.addEventListener("cancel", function (event) {
    event.preventDefault();
    close();
  });
  dialog.addEventListener("click", function (event) {
    if (event.target === dialog || event.target === shell || event.target === stage) close();
  });
  dialog.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      show(index - 1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      show(index + 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      show(0);
    } else if (event.key === "End") {
      event.preventDefault();
      show(images.length - 1);
    }
  });

  stage.addEventListener("touchstart", function (event) {
    if (event.touches.length !== 1) {
      touchStart = null;
      return;
    }
    touchStart = { x: event.touches[0].clientX, y: event.touches[0].clientY };
  }, { passive: true });
  stage.addEventListener("touchend", function (event) {
    if (!touchStart || event.changedTouches.length !== 1) return;
    var dx = event.changedTouches[0].clientX - touchStart.x;
    var dy = event.changedTouches[0].clientY - touchStart.y;
    touchStart = null;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.25) {
      show(index + (dx < 0 ? 1 : -1));
    }
  }, { passive: true });
})();
