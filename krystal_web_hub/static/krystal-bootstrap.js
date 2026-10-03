/**
 * ============================================================================
 * Krystal-Bootstrap: Web Component Runtime & Declarative 3D Engine Library
 * ============================================================================
 * The 3D/Procedural Engine Analog of Twitter Bootstrap for Web Technologies.
 * Registers custom HTML elements: <krystal-viewport>, <krystal-scene>, <krystal-instance>
 */

(function (window, document) {
  'use strict';

  class KrystalViewport extends HTMLElement {
    constructor() {
      super();
      this.attachShadow({ mode: 'open' });
      this.mode = this.getAttribute('mode') || 'CYBERPUNK';
      this.streamUrl = this.getAttribute('src') || '/api/stream';
      this.cols = parseInt(this.getAttribute('cols') || '72', 10);
      this.rows = parseInt(this.getAttribute('rows') || '30', 10);
      this.fx = this.getAttribute('fx') || 'scanlines';
      this.eventSource = null;
      this.activeAscii = 'KRYSTAL-BOOTSTRAP INITIALIZING...';
    }

    connectedCallback() {
      this.renderTemplate();
      this.connectStream();
    }

    disconnectedCallback() {
      if (this.eventSource) {
        this.eventSource.close();
      }
    }

    renderTemplate() {
      const style = `
        :host {
          display: block;
          position: relative;
          background: #07090e;
          border: 1px solid rgba(0, 240, 255, 0.3);
          border-radius: 8px;
          overflow: hidden;
          font-family: 'Fira Code', 'Courier New', monospace;
          box-shadow: 0 8px 30px rgba(0,0,0,0.6);
        }
        .k-topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 8px 12px;
          background: rgba(14, 20, 32, 0.9);
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
          font-size: 0.72rem;
          color: #00f0ff;
          letter-spacing: 1px;
        }
        .k-topbar .tag {
          background: rgba(0, 240, 255, 0.15);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 0.65rem;
        }
        .k-screen {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 16px;
          background: radial-gradient(circle at center, #0f1624 0%, #05070a 100%);
          min-height: 240px;
          overflow: hidden;
        }
        pre.k-ascii {
          margin: 0;
          font-size: 10px;
          line-height: 1.15;
          letter-spacing: 0.5px;
          white-space: pre;
          color: #00f0ff;
          text-shadow: 0 0 6px rgba(0, 240, 255, 0.4);
          user-select: none;
        }
        .scanlines::after {
          content: ' ';
          position: absolute;
          top: 0; left: 0; right: 0; bottom: 0;
          background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.3) 50%);
          background-size: 100% 3px;
          pointer-events: none;
        }
      `;

      this.shadowRoot.innerHTML = `
        <style>${style}</style>
        <div class="k-topbar">
          <div><span style="color:#ff0055;">●</span> KRYSTAL-VIEWPORT [${this.mode}]</div>
          <div class="tag" id="fpsTag">30.0 FPS</div>
        </div>
        <div class="k-screen ${this.fx.includes('scanlines') ? 'scanlines' : ''}">
          <pre class="k-ascii" id="asciiView">${this.activeAscii}</pre>
        </div>
      `;
    }

    connectStream() {
      try {
        this.eventSource = new EventSource(this.streamUrl);
        this.eventSource.addEventListener('frame', (e) => {
          try {
            const data = JSON.parse(e.data);
            if (data.ascii) {
              const view = this.shadowRoot.getElementById('asciiView');
              if (view) view.textContent = data.ascii;
            }
            if (data.fps) {
              const tag = this.shadowRoot.getElementById('fpsTag');
              if (tag) tag.textContent = `${data.fps.toFixed(1)} FPS`;
            }
          } catch (err) {}
        });
      } catch (err) {
        console.warn('KrystalViewport SSE stream fallback:', err);
      }
    }
  }

  class KrystalScene extends HTMLElement {
    constructor() { super(); }
  }

  class KrystalGrid extends HTMLElement {
    constructor() { super(); }
  }

  class KrystalInstance extends HTMLElement {
    constructor() { super(); }
  }

  class KrystalModifier extends HTMLElement {
    constructor() { super(); }
  }

  // Register Custom Web Elements
  if (!customElements.get('krystal-viewport')) {
    customElements.define('krystal-viewport', KrystalViewport);
  }
  if (!customElements.get('krystal-scene')) {
    customElements.define('krystal-scene', KrystalScene);
  }
  if (!customElements.get('krystal-grid')) {
    customElements.define('krystal-grid', KrystalGrid);
  }
  if (!customElements.get('krystal-instance')) {
    customElements.define('krystal-instance', KrystalInstance);
  }
  if (!customElements.get('krystal-modifier')) {
    customElements.define('krystal-modifier', KrystalModifier);
  }

  // Global Engine Bootstrap API
  window.KrystalBootstrap = {
    version: '1.0.0',
    init: function () {
      console.log('[KRYSTAL-BOOTSTRAP] Web Engine Bootstrap initialized. Custom 3D elements active.');
    },
    generateInstance: async function (category, seed) {
      const res = await fetch('/api/instances/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: category, seed: seed })
      });
      return await res.json();
    },
    fetchSchema: async function () {
      const res = await fetch('/api/bootstrap/schema');
      return await res.json();
    }
  };

  document.addEventListener('DOMContentLoaded', () => {
    window.KrystalBootstrap.init();
  });

})(window, document);
