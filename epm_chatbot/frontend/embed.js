/*! EPM Chat embed loader.
 *
 * Opt-in only: it does nothing unless the <script> tag that loads it carries the
 * `data-epm-chat` attribute. It never installs itself anywhere on its own.
 *
 *   <script src="https://chat.example.com/embed.js" data-epm-chat defer></script>
 *
 * Optional attributes:
 *   data-mode="widget" (default, floating button + panel) | "inline" (fills data-target)
 *   data-target="#chat"          element to fill in inline mode (host controls its size)
 *   data-position="right"|"left" widget corner
 *   data-label="Ask EPM"         launcher text
 *   data-theme="light"|"dark"    initial theme
 *
 * Isolation: the chat runs in an iframe on its own origin; the launcher lives in a
 * closed shadow root, so host CSS cannot restyle it and its CSS cannot leak out.
 * The server only lets allow-listed origins frame the chat (CSP frame-ancestors) and
 * this loader first asks the server whether the current site is permitted.
 */
(function () {
  "use strict";
  var script = document.currentScript;
  if (!script || !script.hasAttribute("data-epm-chat")) return;   // explicit opt-in
  if (window.__epmChatLoaded) return;
  window.__epmChatLoaded = true;

  var chatOrigin;
  try { chatOrigin = new URL(script.src, location.href).origin; } catch (e) { return; }
  var mode = script.getAttribute("data-mode") === "inline" ? "inline" : "widget";
  var position = script.getAttribute("data-position") === "left" ? "left" : "right";
  var label = (script.getAttribute("data-label") || "Ask EPM").slice(0, 40);
  var theme = script.getAttribute("data-theme");
  theme = theme === "dark" || theme === "light" ? theme : "";

  function frameSrc() {
    var u = new URL("/", chatOrigin);
    u.searchParams.set("embed", "1");
    u.searchParams.set("parent", location.origin);
    if (theme) u.searchParams.set("theme", theme);
    return u.toString();
  }

  function makeFrame() {
    var f = document.createElement("iframe");
    f.src = frameSrc();
    f.title = "EPM Chat";
    f.setAttribute("allow", "clipboard-write");
    f.setAttribute("referrerpolicy", "strict-origin");
    f.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms allow-downloads allow-popups");
    return f;
  }

  function listen(frame, handlers) {
    window.addEventListener("message", function (e) {
      if (e.origin !== chatOrigin || e.source !== frame.contentWindow) return;   // strict origin + source check
      var d = e.data || {};
      if (typeof d.type === "string" && handlers[d.type]) handlers[d.type](d);
    });
  }

  function inline() {
    var sel = script.getAttribute("data-target");
    var target = sel ? document.querySelector(sel) : null;
    if (!target) { console.warn("[EPM Chat] data-target not found:", sel); return; }
    var f = makeFrame();
    f.style.cssText = "width:100%;height:100%;min-height:420px;border:0;display:block;";
    target.appendChild(f);
  }

  function widget() {
    var host = document.createElement("div");
    host.setAttribute("data-epm-chat-host", "");
    var root = host.attachShadow({ mode: "closed" });
    var side = position === "left" ? "left" : "right";
    var style = document.createElement("style");
    style.textContent = [
      ":host{all:initial}",
      ".launch{position:fixed;bottom:20px;" + side + ":20px;z-index:2147483000;display:inline-flex;align-items:center;gap:8px;",
      "padding:12px 18px;border:0;border-radius:999px;background:#126b5e;color:#fff;font:600 14px/1 system-ui,-apple-system,'Segoe UI',sans-serif;",
      "box-shadow:0 6px 20px rgba(18,107,94,.28);cursor:pointer}",
      ".launch:focus-visible{outline:3px solid #2f6fe0;outline-offset:3px}",
      ".launch svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}",
      ".panel{position:fixed;bottom:84px;" + side + ":20px;z-index:2147483000;width:min(420px,calc(100vw - 24px));",
      "height:min(680px,calc(100vh - 110px));border-radius:16px;overflow:hidden;background:#f7f8f6;",
      "box-shadow:0 18px 50px rgba(0,0,0,.22);border:1px solid rgba(0,0,0,.08)}",
      ".panel[hidden]{display:none}",
      ".panel iframe{width:100%;height:100%;border:0;display:block}",
      "@media (max-width:520px){.panel{inset:0;width:100vw;height:100dvh;border-radius:0;bottom:0;" + side + ":0}}"
    ].join("");
    var btn = document.createElement("button");
    btn.className = "launch";
    btn.type = "button";
    btn.setAttribute("aria-expanded", "false");
    btn.setAttribute("aria-controls", "epm-chat-panel");
    btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5h16v11H9l-5 4z"/></svg>';
    btn.appendChild(document.createTextNode(label));
    var panel = document.createElement("div");
    panel.className = "panel"; panel.id = "epm-chat-panel"; panel.hidden = true;
    panel.setAttribute("role", "dialog"); panel.setAttribute("aria-label", "EPM Chat");
    var frame = null;

    function open() {
      if (!frame) { frame = makeFrame(); panel.appendChild(frame); listen(frame, handlers); }
      panel.hidden = false; btn.setAttribute("aria-expanded", "true");
      if (frame.contentWindow) frame.contentWindow.postMessage({ type: "epm-chat:focus" }, chatOrigin);
    }
    function close() { panel.hidden = true; btn.setAttribute("aria-expanded", "false"); btn.focus(); }
    var handlers = {
      "epm-chat:close": close,
      "epm-chat:ready": function () { if (!panel.hidden) frame.contentWindow.postMessage({ type: "epm-chat:focus" }, chatOrigin); }
    };
    btn.addEventListener("click", function () { panel.hidden ? open() : close(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && !panel.hidden) close(); });
    root.appendChild(style); root.appendChild(panel); root.appendChild(btn);
    document.body.appendChild(host);
  }

  function start() { (mode === "inline" ? inline : widget)(); }

  // Ask the chat server whether this website is on its allowlist before showing anything.
  fetch(chatOrigin + "/embed/status", { mode: "cors", credentials: "omit", cache: "no-store" })
    .then(function (r) { if (!r.ok) throw new Error("not allowed"); return r.json(); })
    .then(function (j) {
      if (!j.allowed) throw new Error("not allowed");
      if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
    })
    .catch(function () {
      console.warn("[EPM Chat] " + location.origin + " is not on this chat server's allowlist (EPM_CHAT_ALLOWED_ORIGINS); the widget was not loaded.");
    });
})();
