document$.subscribe(function () {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: false,
      theme: document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "default",
      securityLevel: "loose",
    });
    mermaid.run({ querySelector: ".mermaid" });
  }
});
