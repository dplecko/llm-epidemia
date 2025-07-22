(function highlightActiveLink() {
  const current = window.location.pathname.split("/").pop();

  const tryHighlight = () => {
    const links = document.querySelectorAll(".nav-link-btn");
    if (links.length === 0) return;

    links.forEach(link => {
      if (link.dataset.page === current) {
        link.classList.add("active-page");
      }
    });

    clearInterval(loop);
  };

  const loop = setInterval(tryHighlight, 100);
})();
