/**
 * KRYSTAL-STACK // WEBOS WINDOW MANAGEMENT ENGINE
 * Features: Floating, Draggable, Resizable, Dockable, Minimizable/Maximizable
 * with Edge Snapping, Z-Index focus layering, and Taskbar integration.
 * Invariant: VITAL_MAX_HP = 6
 */

class KrystalWindowManager {
  constructor(options = {}) {
    this.desktopEl = document.querySelector(options.desktopSelector || '.k-desktop');
    this.taskbarItemsEl = document.querySelector(options.taskbarSelector || '.k-taskbar-items');
    this.snapPreviewEl = document.querySelector(options.snapPreviewSelector || '.k-snap-preview');
    this.startMenuEl = document.querySelector(options.startMenuSelector || '.k-start-menu');
    this.startBtnEl = document.querySelector(options.startBtnSelector || '.k-start-btn');
    
    this.windows = new Map();
    this.activeWindowId = null;
    this.baseZIndex = 100;
    this.highestZIndex = 100;
    this.snapThreshold = 25; // px from edge
    this.currentSnap = null; // 'full' | 'left' | 'right' | null

    this._initGlobalListeners();
    this._initClock();
    this._initStartMenu();
  }

  _initGlobalListeners() {
    // Snap preview helper
    if (!this.snapPreviewEl && this.desktopEl) {
      this.snapPreviewEl = document.createElement('div');
      this.snapPreviewEl.className = 'k-snap-preview';
      this.desktopEl.appendChild(this.snapPreviewEl);
    }

    // Deselect desktop icons when clicking empty desktop
    this.desktopEl.addEventListener('click', (e) => {
      if (e.target === this.desktopEl || e.target.classList.contains('k-desktop-scanlines')) {
        document.querySelectorAll('.k-desktop-icon').forEach(icon => icon.classList.remove('selected'));
        if (this.startMenuEl) this.startMenuEl.classList.remove('open');
      }
    });

    // Close start menu when clicking outside
    document.addEventListener('click', (e) => {
      if (this.startMenuEl && this.startMenuEl.classList.contains('open')) {
        if (!this.startMenuEl.contains(e.target) && !this.startBtnEl.contains(e.target)) {
          this.startMenuEl.classList.remove('open');
        }
      }
    });
  }

  _initStartMenu() {
    if (this.startBtnEl && this.startMenuEl) {
      this.startBtnEl.addEventListener('click', (e) => {
        e.stopPropagation();
        this.startMenuEl.classList.toggle('open');
      });

      // Filter search
      const searchInput = this.startMenuEl.querySelector('input');
      if (searchInput) {
        searchInput.addEventListener('input', (e) => {
          const query = e.target.value.toLowerCase();
          const items = this.startMenuEl.querySelectorAll('.k-start-app-item');
          items.forEach(item => {
            const text = item.innerText.toLowerCase();
            item.style.display = text.includes(query) ? 'flex' : 'none';
          });
        });
      }
    }
  }

  _initClock() {
    const clockEl = document.getElementById('trayClock');
    const updateTime = () => {
      if (clockEl) {
        const now = new Date();
        const hrs = String(now.getHours()).padStart(2, '0');
        const mins = String(now.getMinutes()).padStart(2, '0');
        const secs = String(now.getSeconds()).padStart(2, '0');
        clockEl.innerText = `${hrs}:${mins}:${secs}`;
      }
    };
    updateTime();
    setInterval(updateTime, 1000);
  }

  createWindow(cfg) {
    const {
      id,
      title = 'Krystal Window',
      icon = '💎',
      badge = 'CORE',
      width = 500,
      height = 400,
      x = 50 + (this.windows.size * 30) % 200,
      y = 50 + (this.windows.size * 25) % 150,
      minWidth = 340,
      minHeight = 220,
      content = '',
      isIframe = false,
      url = '',
      noPadding = false,
      onInit = null
    } = cfg;

    if (this.windows.has(id)) {
      this.restoreWindow(id);
      this.focusWindow(id);
      return this.windows.get(id);
    }

    // 1. Create Window DOM
    const win = document.createElement('div');
    win.className = 'k-window';
    win.id = `win-${id}`;
    win.style.width = `${width}px`;
    win.style.height = `${height}px`;
    win.style.left = `${x}px`;
    win.style.top = `${y}px`;
    win.style.zIndex = ++this.highestZIndex;

    win.innerHTML = `
      <div class="k-window-header">
        <div class="k-window-title">
          <span class="k-window-icon">${icon}</span>
          <span class="k-window-name">${title}</span>
          <span class="k-window-badge">${badge}</span>
        </div>
        <div class="k-window-controls">
          <button class="k-window-btn minimize" title="Minimalizovať" data-action="minimize">_</button>
          <button class="k-window-btn maximize" title="Maximalizovať / Obnoviť" data-action="maximize">□</button>
          <button class="k-window-btn close" title="Zavrieť" data-action="close">✕</button>
        </div>
      </div>
      <div class="k-window-body ${noPadding ? 'no-padding' : ''}" id="body-${id}">
        ${isIframe ? `<iframe src="${url}" class="k-window-iframe"></iframe>` : content}
      </div>
      <!-- Resize handles -->
      <div class="k-resize-handle n" data-dir="n"></div>
      <div class="k-resize-handle s" data-dir="s"></div>
      <div class="k-resize-handle e" data-dir="e"></div>
      <div class="k-resize-handle w" data-dir="w"></div>
      <div class="k-resize-handle ne" data-dir="ne"></div>
      <div class="k-resize-handle nw" data-dir="nw"></div>
      <div class="k-resize-handle se" data-dir="se"></div>
      <div class="k-resize-handle sw" data-dir="sw"></div>
    `;

    this.desktopEl.appendChild(win);

    // 2. Create Taskbar Item
    const taskItem = document.createElement('div');
    taskItem.className = 'k-taskbar-item active';
    taskItem.id = `task-${id}`;
    taskItem.innerHTML = `<span>${icon}</span> <span>${title}</span>`;
    taskItem.addEventListener('click', () => {
      const winData = this.windows.get(id);
      if (winData.isMinimized) {
        this.restoreWindow(id);
        this.focusWindow(id);
      } else if (this.activeWindowId === id) {
        this.minimizeWindow(id);
      } else {
        this.focusWindow(id);
      }
    });
    this.taskbarItemsEl.appendChild(taskItem);

    const winRecord = {
      id,
      el: win,
      taskEl: taskItem,
      title,
      icon,
      isMinimized: false,
      isMaximized: false,
      prevBounds: { x, y, width, height },
      minWidth,
      minHeight
    };

    this.windows.set(id, winRecord);

    // 3. Attach Window Events
    this._attachWindowInteractions(winRecord);

    // 4. Focus
    this.focusWindow(id);

    if (typeof onInit === 'function') {
      try {
        onInit(win.querySelector(`#body-${id}`), winRecord);
      } catch (err) {
        console.error(`Error in onInit for window ${id}:`, err);
      }
    }

    return winRecord;
  }

  _attachWindowInteractions(w) {
    const { el, id } = w;
    const header = el.querySelector('.k-window-header');

    // Focus on click
    el.addEventListener('mousedown', () => {
      this.focusWindow(id);
    });

    // Control buttons
    header.addEventListener('click', (e) => {
      const btn = e.target.closest('.k-window-btn');
      if (!btn) return;
      const action = btn.dataset.action;
      if (action === 'minimize') this.minimizeWindow(id);
      if (action === 'maximize') this.toggleMaximizeWindow(id);
      if (action === 'close') this.closeWindow(id);
    });

    // Double-click titlebar to toggle maximize
    header.addEventListener('dblclick', (e) => {
      if (e.target.closest('.k-window-controls')) return;
      this.toggleMaximizeWindow(id);
    });

    // Dragging
    header.addEventListener('mousedown', (e) => {
      if (e.target.closest('.k-window-controls')) return;
      if (w.isMaximized) return; // don't drag if maximized

      this.focusWindow(id);
      let startX = e.clientX;
      let startY = e.clientY;
      let initialLeft = el.offsetLeft;
      let initialTop = el.offsetTop;

      const onMouseMove = (moveEvent) => {
        const dx = moveEvent.clientX - startX;
        const dy = moveEvent.clientY - startY;

        let newLeft = initialLeft + dx;
        let newTop = initialTop + dy;

        // Snap indicator detection
        const deskRect = this.desktopEl.getBoundingClientRect();
        this.currentSnap = null;

        if (moveEvent.clientY <= deskRect.top + this.snapThreshold) {
          // Snap Full Screen
          this.currentSnap = 'full';
          this.showSnapPreview(0, 0, deskRect.width, deskRect.height);
        } else if (moveEvent.clientX <= deskRect.left + this.snapThreshold) {
          // Snap Left 50%
          this.currentSnap = 'left';
          this.showSnapPreview(0, 0, deskRect.width / 2, deskRect.height);
        } else if (moveEvent.clientX >= deskRect.right - this.snapThreshold) {
          // Snap Right 50%
          this.currentSnap = 'right';
          this.showSnapPreview(deskRect.width / 2, 0, deskRect.width / 2, deskRect.height);
        } else {
          this.hideSnapPreview();
        }

        // Keep inside desktop top boundary
        if (newTop < 0) newTop = 0;

        el.style.left = `${newLeft}px`;
        el.style.top = `${newTop}px`;
      };

      const onMouseUp = () => {
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('mouseup', onMouseUp);

        if (this.currentSnap) {
          const deskRect = this.desktopEl.getBoundingClientRect();
          if (this.currentSnap === 'full') {
            this.maximizeWindow(id);
          } else if (this.currentSnap === 'left') {
            this.snapWindow(id, 0, 0, deskRect.width / 2, deskRect.height);
          } else if (this.currentSnap === 'right') {
            this.snapWindow(id, deskRect.width / 2, 0, deskRect.width / 2, deskRect.height);
          }
          this.hideSnapPreview();
          this.currentSnap = null;
        } else {
          w.prevBounds.x = el.offsetLeft;
          w.prevBounds.y = el.offsetTop;
        }
      };

      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    });

    // Resizing
    const handles = el.querySelectorAll('.k-resize-handle');
    handles.forEach(handle => {
      handle.addEventListener('mousedown', (e) => {
        e.stopPropagation();
        if (w.isMaximized) return;

        this.focusWindow(id);
        const dir = handle.dataset.dir;
        let startX = e.clientX;
        let startY = e.clientY;
        let startWidth = el.offsetWidth;
        let startHeight = el.offsetHeight;
        let startLeft = el.offsetLeft;
        let startTop = el.offsetTop;

        const onResizeMove = (moveEvent) => {
          const dx = moveEvent.clientX - startX;
          const dy = moveEvent.clientY - startY;

          if (dir.includes('e')) {
            const newW = Math.max(w.minWidth, startWidth + dx);
            el.style.width = `${newW}px`;
          }
          if (dir.includes('s')) {
            const newH = Math.max(w.minHeight, startHeight + dy);
            el.style.height = `${newH}px`;
          }
          if (dir.includes('w')) {
            const newW = Math.max(w.minWidth, startWidth - dx);
            if (newW > w.minWidth) {
              el.style.width = `${newW}px`;
              el.style.left = `${startLeft + dx}px`;
            }
          }
          if (dir.includes('n')) {
            const newH = Math.max(w.minHeight, startHeight - dy);
            if (newH > w.minHeight) {
              el.style.height = `${newH}px`;
              el.style.top = `${startTop + dy}px`;
            }
          }
        };

        const onResizeUp = () => {
          document.removeEventListener('mousemove', onResizeMove);
          document.removeEventListener('mouseup', onResizeUp);
          w.prevBounds.width = el.offsetWidth;
          w.prevBounds.height = el.offsetHeight;
          w.prevBounds.x = el.offsetLeft;
          w.prevBounds.y = el.offsetTop;
        };

        document.addEventListener('mousemove', onResizeMove);
        document.addEventListener('mouseup', onResizeUp);
      });
    });
  }

  showSnapPreview(x, y, w, h) {
    if (!this.snapPreviewEl) return;
    this.snapPreviewEl.style.display = 'block';
    this.snapPreviewEl.style.left = `${x}px`;
    this.snapPreviewEl.style.top = `${y}px`;
    this.snapPreviewEl.style.width = `${w}px`;
    this.snapPreviewEl.style.height = `${h}px`;
  }

  hideSnapPreview() {
    if (this.snapPreviewEl) this.snapPreviewEl.style.display = 'none';
  }

  snapWindow(id, x, y, w, h) {
    const win = this.windows.get(id);
    if (!win) return;
    win.el.style.left = `${x}px`;
    win.el.style.top = `${y}px`;
    win.el.style.width = `${w}px`;
    win.el.style.height = `${h}px`;
    win.isMaximized = false;
    win.el.classList.remove('maximized');
  }

  focusWindow(id) {
    const win = this.windows.get(id);
    if (!win) return;

    this.activeWindowId = id;
    this.highestZIndex++;
    win.el.style.zIndex = this.highestZIndex;

    this.windows.forEach((w, wId) => {
      if (wId === id) {
        w.el.classList.add('focused');
        w.taskEl.classList.add('active');
      } else {
        w.el.classList.remove('focused');
        w.taskEl.classList.remove('active');
      }
    });
  }

  minimizeWindow(id) {
    const win = this.windows.get(id);
    if (!win) return;
    win.isMinimized = true;
    win.el.classList.add('minimized');
    win.el.classList.remove('focused');
    win.taskEl.classList.add('minimized');
    win.taskEl.classList.remove('active');

    // Focus next available window
    if (this.activeWindowId === id) {
      this.activeWindowId = null;
      let nextWin = null;
      let maxZ = -1;
      this.windows.forEach(w => {
        if (!w.isMinimized && parseInt(w.el.style.zIndex) > maxZ) {
          maxZ = parseInt(w.el.style.zIndex);
          nextWin = w;
        }
      });
      if (nextWin) this.focusWindow(nextWin.id);
    }
  }

  restoreWindow(id) {
    const win = this.windows.get(id);
    if (!win) return;
    win.isMinimized = false;
    win.el.classList.remove('minimized');
    win.taskEl.classList.remove('minimized');
    this.focusWindow(id);
  }

  toggleMaximizeWindow(id) {
    const win = this.windows.get(id);
    if (!win) return;
    if (win.isMaximized) {
      this.unmaximizeWindow(id);
    } else {
      this.maximizeWindow(id);
    }
  }

  maximizeWindow(id) {
    const win = this.windows.get(id);
    if (!win) return;

    // Save previous bounds
    win.prevBounds = {
      x: win.el.offsetLeft,
      y: win.el.offsetTop,
      width: win.el.offsetWidth,
      height: win.el.offsetHeight
    };

    const deskRect = this.desktopEl.getBoundingClientRect();
    win.el.style.left = '0px';
    win.el.style.top = '0px';
    win.el.style.width = `${deskRect.width}px`;
    win.el.style.height = `${deskRect.height}px`;
    win.isMaximized = true;
    win.el.classList.add('maximized');
    this.focusWindow(id);
  }

  unmaximizeWindow(id) {
    const win = this.windows.get(id);
    if (!win) return;

    win.el.style.left = `${win.prevBounds.x}px`;
    win.el.style.top = `${win.prevBounds.y}px`;
    win.el.style.width = `${win.prevBounds.width}px`;
    win.el.style.height = `${win.prevBounds.height}px`;
    win.isMaximized = false;
    win.el.classList.remove('maximized');
    this.focusWindow(id);
  }

  closeWindow(id) {
    const win = this.windows.get(id);
    if (!win) return;

    win.el.remove();
    win.taskEl.remove();
    this.windows.delete(id);

    if (this.activeWindowId === id) {
      this.activeWindowId = null;
      let nextWin = null;
      let maxZ = -1;
      this.windows.forEach(w => {
        if (!w.isMinimized && parseInt(w.el.style.zIndex) > maxZ) {
          maxZ = parseInt(w.el.style.zIndex);
          nextWin = w;
        }
      });
      if (nextWin) this.focusWindow(nextWin.id);
    }
  }
}
