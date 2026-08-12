"""Ambient, pointer-reactive background layer for the PromoStrater dashboard.

Streamlit renders custom components inside a sandboxed iframe, which is the
wrong shape for an effect that has to span the whole viewport and read the
pointer everywhere on the page. Because the component iframe is same-origin
with the app, this module uses a zero-height component purely as a script
host: it reaches ``window.parent.document`` and mounts a single fixed canvas
*behind* Streamlit's own DOM, plus a pointer-tracked sheen on the cards.

Three layers, all driven by one rAF loop:

``aurora``    slow-drifting gradient orbs that lean toward the pointer
``field``     a particle grid that parts around the cursor and drifts back
``halo``      a soft spotlight trailing the cursor with easing

Colours come from the active palette in :mod:`dashboard.palettes`, so the
background follows whichever scheme is selected instead of pinning its own
hues. The whole effect is skipped under ``prefers-reduced-motion``, matching
the guard that already closes ``theme.py``.
"""

from __future__ import annotations

import streamlit as st

from dashboard.theme import selected_palette

_AMBIENT_JS = """
<script>
(function () {
  var win, doc;
  try { win = window.parent; doc = win.document; } catch (e) { return; }
  if (!doc || !doc.body) return;

  // Streamlit re-executes the script top-to-bottom on every rerun, which
  // remounts this component. Tear the previous instance down rather than
  // stacking a second canvas and a second rAF loop on top of the first.
  if (win.__psAmbientTeardown) { try { win.__psAmbientTeardown(); } catch (e) {} }

  // The host iframe is only a script carrier; keep it out of the layout.
  try {
    // Not `frame`: that name belongs to the rAF callback declared below, and
    // a `var` here would shadow the hoisted function in this same scope.
    var hostFrame = window.frameElement;
    if (hostFrame) {
      var slot = hostFrame.closest('[data-testid="stElementContainer"]') || hostFrame.parentElement;
      if (slot) slot.style.display = 'none';
    }
  } catch (e) {}

  var reduced = win.matchMedia
    && win.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------------------------------------------------------- *
   * Stacking + pointer sheen
   *
   * .ps-kpi-card already spends both ::before and ::after (theme.py),
   * so the sheen rides on background-image instead.
   * ---------------------------------------------------------------- */
  var STYLE_ID = 'ps-ambient-style';
  var stale = doc.getElementById(STYLE_ID);
  if (stale) stale.remove();

  // Parks off-canvas until a real pointer move sets --ps-mx/--ps-my.
  var SHEEN = 'radial-gradient(260px circle at var(--ps-mx,-999px) var(--ps-my,-999px),'
            + 'rgba(__TONE_1__,0.15),rgba(__TONE_2__,0.09) 42%,transparent 68%)';

  var style = doc.createElement('style');
  style.id = STYLE_ID;
  style.textContent = [
    '#ps-ambient{position:fixed;inset:0;z-index:0;pointer-events:none;}',
    '.stApp,.stAppViewContainer,.stMain,[data-testid="stMain"]',
    '{background:transparent !important;}',
    '.stApp > *{position:relative;z-index:1;}',
    // theme.py is injected into the body, so it beats a head rule on document
    // order; !important is what lets the sheen land. .ps-kpi-card already owns
    // a background-image, so its gradient is restated as the lower layer
    // instead of being clobbered.
    '.ps-card,.ps-step-card{background-image:' + SHEEN + ' !important;}',
    '.ps-kpi-card{background-image:' + SHEEN + ',',
    '  var(--ps-glass-grad) !important;}'
  ].join('\\n');
  doc.head.appendChild(style);

  var SHEEN_SELECTOR = '.ps-kpi-card, .ps-card, .ps-step-card';

  /* ---------------------------------------------------------------- *
   * Canvas
   * ---------------------------------------------------------------- */
  var prior = doc.getElementById('ps-ambient');
  if (prior) prior.remove();

  var canvas = doc.createElement('canvas');
  canvas.id = 'ps-ambient';
  doc.body.insertBefore(canvas, doc.body.firstChild);
  var ctx = canvas.getContext('2d');

  var W = 0, H = 0, dpr = 1;
  function resize() {
    dpr = Math.min(win.devicePixelRatio || 1, 2);
    W = win.innerWidth; H = win.innerHeight;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seedField();
  }

  /* Aurora orbs. Each drifts on its own slow sine and leans toward the
     pointer by `pull`, so the background feels like it notices you. */
  var ORBS = [
    { x: 0.18, y: 0.22, r: 0.42, c: '__TONE_1__',  a: 0.20, sx: 0.00021, sy: 0.00017, pull: 0.055 },
    { x: 0.82, y: 0.30, r: 0.36, c: '__TONE_2__', a: 0.16, sx: 0.00017, sy: 0.00024, pull: 0.085 },
    { x: 0.62, y: 0.78, r: 0.44, c: '__TONE_3__',   a: 0.13, sx: 0.00013, sy: 0.00019, pull: 0.040 },
    { x: 0.30, y: 0.86, r: 0.30, c: '__TONE_1__',  a: 0.11, sx: 0.00025, sy: 0.00012, pull: 0.070 }
  ];

  var field = [];
  function seedField() {
    field = [];
    // Density scales with viewport so a laptop is not doing 4K particle work.
    var step = 74;
    for (var gx = step * 0.5; gx < W; gx += step) {
      for (var gy = step * 0.5; gy < H; gy += step) {
        var jx = (Math.random() - 0.5) * step * 0.55;
        var jy = (Math.random() - 0.5) * step * 0.55;
        field.push({
          hx: gx + jx, hy: gy + jy,           // home
          x: gx + jx,  y: gy + jy,            // current
          vx: 0, vy: 0,
          r: 0.9 + Math.random() * 1.5,
          depth: 0.35 + Math.random() * 0.9   // parallax weight
        });
      }
    }
  }

  /* ---------------------------------------------------------------- *
   * Pointer + scroll
   * ---------------------------------------------------------------- */
  var px = -9999, py = -9999;      // raw pointer
  var hx = -9999, hy = -9999;      // eased halo, trails the pointer
  var hasPointer = false;
  var scrollY = 0, scrollEased = 0;

  function onMove(e) {
    px = e.clientX; py = e.clientY;
    if (!hasPointer) { hx = px; hy = py; hasPointer = true; }

    // Feed the card sheen. elementsFromPoint is cheap enough at pointer rate
    // and avoids attaching a listener per card as Streamlit rebuilds the DOM.
    var stack = doc.elementsFromPoint(px, py) || [];
    for (var i = 0; i < stack.length; i++) {
      var el = stack[i].closest ? stack[i].closest(SHEEN_SELECTOR) : null;
      if (el) {
        var b = el.getBoundingClientRect();
        el.style.setProperty('--ps-mx', (px - b.left) + 'px');
        el.style.setProperty('--ps-my', (py - b.top) + 'px');
        break;
      }
    }
  }
  function onLeave() { px = -9999; py = -9999; hasPointer = false; }

  // Streamlit scrolls an inner container, not the window.
  var scroller = doc.querySelector('.stMain')
    || doc.querySelector('[data-testid="stMain"]')
    || doc.querySelector('section.main');
  function readScroll() {
    scrollY = scroller ? scroller.scrollTop : (win.scrollY || 0);
  }

  doc.addEventListener('pointermove', onMove, { passive: true });
  doc.addEventListener('pointerleave', onLeave, { passive: true });
  win.addEventListener('resize', resize, { passive: true });
  if (scroller) scroller.addEventListener('scroll', readScroll, { passive: true });
  else win.addEventListener('scroll', readScroll, { passive: true });

  /* ---------------------------------------------------------------- *
   * Frame loop
   * ---------------------------------------------------------------- */
  var raf = 0, t0 = null, running = true;

  function orbLayer(t) {
    var cx = hasPointer ? hx : W * 0.5;
    var cy = hasPointer ? hy : H * 0.5;
    for (var i = 0; i < ORBS.length; i++) {
      var o = ORBS[i];
      var bx = (o.x + Math.sin(t * o.sx + i * 1.7) * 0.06) * W;
      var by = (o.y + Math.cos(t * o.sy + i * 2.3) * 0.06) * H;
      // Lean toward the pointer, and drift on scroll for parallax depth.
      var x = bx + (cx - bx) * o.pull;
      var y = by + (cy - by) * o.pull - scrollEased * (0.06 + i * 0.02);
      var r = o.r * Math.max(W, H);

      var g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, 'rgba(' + o.c + ',' + o.a + ')');
      g.addColorStop(0.55, 'rgba(' + o.c + ',' + (o.a * 0.32).toFixed(3) + ')');
      g.addColorStop(1, 'rgba(' + o.c + ',0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, W, H);
    }
  }

  function fieldLayer() {
    var R = 130, R2 = R * R;
    for (var i = 0; i < field.length; i++) {
      var p = field[i];
      var homeY = p.hy - scrollEased * 0.10 * p.depth;

      // Spring home, then push away from the pointer.
      p.vx += (p.hx - p.x) * 0.012;
      p.vy += (homeY - p.y) * 0.012;

      if (hasPointer) {
        var dx = p.x - px, dy = p.y - py;
        var d2 = dx * dx + dy * dy;
        if (d2 < R2 && d2 > 0.01) {
          var d = Math.sqrt(d2);
          var force = (1 - d / R) * 2.6 * p.depth;
          p.vx += (dx / d) * force;
          p.vy += (dy / d) * force;
        }
      }

      p.vx *= 0.86; p.vy *= 0.86;
      p.x += p.vx; p.y += p.vy;

      if (p.y < -40 || p.y > H + 40) continue;
      var speed = Math.min(Math.sqrt(p.vx * p.vx + p.vy * p.vy) / 6, 1);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * (1 + speed * 0.8), 0, 6.2832);
      ctx.fillStyle = 'rgba(__TONE_1__,' + (0.13 + speed * 0.30).toFixed(3) + ')';
      ctx.fill();
    }
  }

  function haloLayer() {
    if (!hasPointer) return;
    var g = ctx.createRadialGradient(hx, hy, 0, hx, hy, 190);
    g.addColorStop(0, 'rgba(__TONE_2__,0.10)');
    g.addColorStop(0.45, 'rgba(__TONE_1__,0.055)');
    g.addColorStop(1, 'rgba(__TONE_1__,0)');
    ctx.fillStyle = g;
    ctx.fillRect(hx - 190, hy - 190, 380, 380);
  }

  function frame(ts) {
    if (!running) return;
    if (t0 === null) t0 = ts;
    var t = ts - t0;

    hx += (px - hx) * 0.11;
    hy += (py - hy) * 0.11;
    scrollEased += (scrollY - scrollEased) * 0.09;

    ctx.clearRect(0, 0, W, H);
    orbLayer(t);
    fieldLayer();
    haloLayer();

    raf = win.requestAnimationFrame(frame);
  }

  function onVisibility() {
    if (doc.hidden) {
      running = false;
      if (raf) win.cancelAnimationFrame(raf);
      raf = 0;
    } else if (!running) {
      running = true; t0 = null;
      raf = win.requestAnimationFrame(frame);
    }
  }
  doc.addEventListener('visibilitychange', onVisibility);

  resize();
  readScroll();
  scrollEased = scrollY;

  // Paint one frame synchronously. rAF does not fire while the tab is in the
  // background, so without this the layer stays blank until first focus.
  ctx.clearRect(0, 0, W, H);
  orbLayer(0);
  fieldLayer();

  if (reduced) {
    running = false;   // static wash only: no loop, no pointer reaction
  } else {
    raf = win.requestAnimationFrame(frame);
  }

  win.__psAmbientTeardown = function () {
    running = false;
    if (raf) win.cancelAnimationFrame(raf);
    doc.removeEventListener('pointermove', onMove);
    doc.removeEventListener('pointerleave', onLeave);
    doc.removeEventListener('visibilitychange', onVisibility);
    win.removeEventListener('resize', resize);
    if (scroller) scroller.removeEventListener('scroll', readScroll);
    else win.removeEventListener('scroll', readScroll);
    var c = doc.getElementById('ps-ambient');
    if (c) c.remove();
    win.__psAmbientTeardown = null;
  };
})();
</script>
"""


def inject_ambient() -> None:
    """Mount the pointer-reactive background layer on the parent document.

    Safe to call once per page immediately after
    :func:`dashboard.theme.inject_theme`. Re-running tears down any previous
    instance first, so Streamlit reruns cannot stack canvases or rAF loops.
    """
    palette = selected_palette()
    payload = (
        _AMBIENT_JS.replace("__TONE_1__", palette["primary-rgb"])
        .replace("__TONE_2__", palette["accent-2-rgb"])
        .replace("__TONE_3__", palette["accent-rgb"])
    )
    # st.iframe supersedes components.v1.html, which warns once per rerun and
    # is removed after 2026-06-01. It keeps the same-origin JS access this
    # layer depends on to reach window.parent. Height must be a *positive*
    # integer here - components.html tolerated 0, st.iframe rejects it - so
    # the frame is 1px and the script hides its own container slot on mount.
    # width is pinned too: if the script ever fails before hiding its slot,
    # a 1px dot is a smaller scar than a full-width 1px rule.
    st.iframe(payload, width=1, height=1)
