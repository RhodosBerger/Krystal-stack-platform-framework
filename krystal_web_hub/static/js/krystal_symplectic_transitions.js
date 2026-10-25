/**
 * krystal_web_hub/static/js/krystal_symplectic_transitions.js
 * ==============================================================================
 * KRYSTAL STACK // SYMPLECTIC ANIMATION & TRANSITION CLIENT LIBRARY
 * ==============================================================================
 * Provides physics-based symplectic spring transitions, particle field rendering,
 * and Hex-HSL chromatic sweeps for ultra-smooth UI/UX interactions.
 * 
 * Constitutional Invariant: VITAL_MAX_HP === 6
 * ==============================================================================
 */

(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.KrystalSymplectic = factory();
    }
}(typeof self !== 'undefined' ? self : this, function () {
    'use strict';

    const VITAL_MAX_HP = 6;

    /**
     * Symplectic Leapfrog Spring Controller
     */
    class SymplecticSpring {
        constructor(config = {}) {
            this.stiffness = config.stiffness || 180.0;
            this.damping = config.damping || 15.0;
            this.mass = config.mass || 1.0;
            this.value = config.initialValue !== undefined ? config.initialValue : 0.0;
            this.target = this.value;
            this.velocity = 0.0;
            this.animating = false;
            this.vital_max_hp = VITAL_MAX_HP;
        }

        setTarget(targetVal) {
            this.target = targetVal;
            this.animating = true;
        }

        step(dtSec) {
            if (!this.animating) return this.value;

            // Clamping dt to avoid large simulation skips
            const dt = Math.min(dtSec, 0.033);

            // Relative coordinate: x = position - target
            let x = this.value - this.target;
            let v = this.velocity;

            // Half-step Leapfrog
            const x_half = x + 0.5 * dt * v;
            const force = -this.stiffness * x_half - this.damping * v;
            const v_next = v + dt * (force / this.mass);
            const x_next = x_half + 0.5 * dt * v_next;

            this.value = this.target + x_next;
            this.velocity = v_next;

            // Check equilibrium termination
            if (Math.abs(x_next) < 0.0005 && Math.abs(v_next) < 0.001) {
                this.value = this.target;
                this.velocity = 0.0;
                this.animating = false;
            }

            return this.value;
        }
    }

    /**
     * Symplectic Interactive Particle Field (HTML5 Canvas)
     */
    class SymplecticParticleCanvas {
        constructor(canvasElement, options = {}) {
            this.canvas = canvasElement;
            this.ctx = canvasElement.getContext('2d');
            this.particleCount = options.particleCount || 48;
            this.particles = [];
            this.width = 0;
            this.height = 0;
            this.mouse = { x: -9999, y: -9999, active: false };
            this.startTime = performance.now();
            this.lastTime = performance.now();
            this.alphaHz = 10.0; // 10 Hz alpha cognitive rhythm
            this.running = false;

            this._resize();
            this._initParticles();
            this._bindEvents();
            this.start();
        }

        _resize() {
            const rect = this.canvas.parentElement ? this.canvas.parentElement.getBoundingClientRect() : { width: window.innerWidth, height: 400 };
            this.width = rect.width;
            this.height = rect.height;
            this.canvas.width = this.width * window.devicePixelRatio;
            this.canvas.height = this.height * window.devicePixelRatio;
            this.canvas.style.width = this.width + 'px';
            this.canvas.style.height = this.height + 'px';
            this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }

        _initParticles() {
            this.particles = [];
            for (let i = 0; i < this.particleCount; i++) {
                this.particles.push({
                    x: Math.random() * this.width,
                    y: Math.random() * this.height,
                    vx: (Math.random() - 0.5) * 0.4,
                    vy: (Math.random() - 0.5) * 0.4,
                    radius: 1.5 + Math.random() * 2.0,
                    hue: (i * 60) % 360,
                    orbitPhase: Math.random() * Math.PI * 2
                });
            }
        }

        _bindEvents() {
            window.addEventListener('resize', () => this._resize());
            this.canvas.addEventListener('mousemove', (e) => {
                const rect = this.canvas.getBoundingClientRect();
                this.mouse.x = e.clientX - rect.left;
                this.mouse.y = e.clientY - rect.top;
                this.mouse.active = true;
            });
            this.canvas.addEventListener('mouseleave', () => {
                this.mouse.active = false;
            });
        }

        start() {
            if (this.running) return;
            this.running = true;
            this.lastTime = performance.now();
            const loop = (now) => {
                if (!this.running) return;
                const dt = (now - this.lastTime) / 1000.0;
                this.lastTime = now;
                this.render(now, dt);
                requestAnimationFrame(loop);
            };
            requestAnimationFrame(loop);
        }

        stop() {
            this.running = false;
        }

        render(now, dt) {
            const elapsed = (now - this.startTime) / 1000.0;
            // 10 Hz alpha harmonic pulse modulation
            const alphaPulse = 0.5 + 0.5 * Math.sin(2 * Math.PI * this.alphaHz * elapsed);

            this.ctx.clearRect(0, 0, this.width, this.height);

            // Update & draw particles
            for (let i = 0; i < this.particles.length; i++) {
                const p = this.particles[i];

                // Orbital drift + Symplectic harmonic jitter
                p.x += p.vx + Math.sin(elapsed * 1.5 + p.orbitPhase) * 0.3;
                p.y += p.vy + Math.cos(elapsed * 1.5 + p.orbitPhase) * 0.3;

                // Mouse interaction spring force
                if (this.mouse.active) {
                    const dx = this.mouse.x - p.x;
                    const dy = this.mouse.y - p.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 120 && dist > 1) {
                        const force = (120 - dist) / 120 * 0.8;
                        p.x -= (dx / dist) * force;
                        p.y -= (dy / dist) * force;
                    }
                }

                // Screen bounce boundaries
                if (p.x < 0) { p.x = 0; p.vx *= -1; }
                if (p.x > this.width) { p.x = this.width; p.vx *= -1; }
                if (p.y < 0) { p.y = 0; p.vy *= -1; }
                if (p.y > this.height) { p.y = this.height; p.vy *= -1; }

                // Dynamic glow based on alpha pulse
                const glowRadius = p.radius * (1.0 + 0.4 * alphaPulse);
                const alpha = 0.4 + 0.5 * alphaPulse;

                this.ctx.beginPath();
                this.ctx.arc(p.x, p.y, glowRadius, 0, Math.PI * 2);
                this.ctx.fillStyle = `hsla(${p.hue}, 90%, 65%, ${alpha})`;
                this.ctx.shadowBlur = 10;
                this.ctx.shadowColor = `hsl(${p.hue}, 90%, 65%)`;
                this.ctx.fill();
                this.ctx.shadowBlur = 0;
            }

            // Draw sympathetic interconnect lines
            for (let i = 0; i < this.particles.length; i++) {
                for (let j = i + 1; j < this.particles.length; j++) {
                    const p1 = this.particles[i];
                    const p2 = this.particles[j];
                    const dx = p1.x - p2.x;
                    const dy = p1.y - p2.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 90) {
                        const lineAlpha = (1.0 - dist / 90) * 0.25 * (0.8 + 0.2 * alphaPulse);
                        this.ctx.beginPath();
                        this.ctx.moveTo(p1.x, p1.y);
                        this.ctx.lineTo(p2.x, p2.y);
                        this.ctx.strokeStyle = `hsla(190, 95%, 60%, ${lineAlpha})`;
                        this.ctx.lineWidth = 1;
                        this.ctx.stroke();
                    }
                }
            }
        }
    }

    /**
     * Smooth Card and View Switcher with Symplectic Spring Physics
     */
    function switchViewWithSymplecticTransition(currentElem, targetElem, options = {}) {
        if (!targetElem) return;
        const duration = options.durationMs || 350;

        if (currentElem) {
            currentElem.style.transition = `opacity ${duration * 0.4}ms ease-out, transform ${duration * 0.6}ms cubic-bezier(0.16, 1, 0.3, 1)`;
            currentElem.style.opacity = '0';
            currentElem.style.transform = 'translateY(-12px) scale(0.98)';
            setTimeout(() => {
                currentElem.style.display = 'none';
                currentElem.style.transform = '';
            }, duration * 0.4);
        }

        targetElem.style.display = options.display || 'block';
        targetElem.style.opacity = '0';
        targetElem.style.transform = 'translateY(16px) scale(0.97)';
        targetElem.style.transition = `opacity ${duration * 0.7}ms ease-out, transform ${duration}ms cubic-bezier(0.34, 1.56, 0.64, 1)`;

        requestAnimationFrame(() => {
            targetElem.style.opacity = '1';
            targetElem.style.transform = 'translateY(0px) scale(1.0)';
        });
    }

    return {
        VITAL_MAX_HP: VITAL_MAX_HP,
        SymplecticSpring: SymplecticSpring,
        SymplecticParticleCanvas: SymplecticParticleCanvas,
        switchViewWithSymplecticTransition: switchViewWithSymplecticTransition
    };
}));
