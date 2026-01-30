// DASHBOARD BUILDER JAVASCRIPT

let selectedComponent = null;
let componentsOnCanvas = [];
let componentIdCounter = 0;

// === DRAG AND DROP ===

function allowDrop(ev) {
    ev.preventDefault();
    document.getElementById('canvas').classList.add('drag-over');
}

function drag(ev) {
    ev.dataTransfer.setData("componentType", ev.target.dataset.component);
}

function drop(ev) {
    ev.preventDefault();
    document.getElementById('canvas').classList.remove('drag-over');

    const componentType = ev.dataTransfer.getData("componentType");
    if (componentType) {
        addComponentToCanvas(componentType, ev.clientX, ev.clientY);
    }
}

// Setup drag handlers
document.addEventListener('DOMContentLoaded', () => {
    const draggables = document.querySelectorAll('.component-item[draggable="true"]');
    draggables.forEach(item => {
        item.addEventListener('dragstart', drag);
    });

    // Remove placeholder when first component added
    const canvas = document.getElementById('canvas');
    const placeholder = canvas.querySelector('.canvas-placeholder');
    if (placeholder) {
        canvas.addEventListener('drop', () => {
            if (placeholder) {
                placeholder.remove();
            }
        }, { once: true });
    }
});

// === ADD COMPONENT TO CANVAS ===

function addComponentToCanvas(componentType) {
    const canvas = document.getElementById('canvas');
    const componentId = `component-${componentIdCounter++}`;

    const componentHTML = generateComponentHTML(componentType, componentId);

    const wrapper = document.createElement('div');
    wrapper.className = 'dropped-component';
    wrapper.id = componentId;
    wrapper.innerHTML = componentHTML;
    wrapper.onclick = () => selectComponent(componentId);

    // Add controls
    const controls = document.createElement('div');
    controls.className = 'component-controls';
    controls.innerHTML = `
        <button onclick="event.stopPropagation(); moveUp('${componentId}')" title="Move Up">↑</button>
        <button onclick="event.stopPropagation(); moveDown('${componentId}')" title="Move Down">↓</button>
        <button onclick="event.stopPropagation(); duplicateComponent('${componentId}')" title="Duplicate">⧉</button>
        <button onclick="event.stopPropagation(); deleteComponent('${componentId}')" title="Delete">🗑️</button>
    `;
    wrapper.appendChild(controls);

    canvas.appendChild(wrapper);

    componentsOnCanvas.push({
        id: componentId,
        type: componentType,
        properties: getDefaultProperties(componentType)
    });

    selectComponent(componentId);
}

// === COMPONENT TEMPLATES ===

function generateComponentHTML(type, id) {
    const templates = {
        'gauge': `
            <div class="component-content">
                <h3>📈 Spindle Load</h3>
                <div style="text-align: center;">
                    <div style="font-size: 3rem; color: #38bdf8;">67%</div>
                    <div class="progress-bar" style="background: rgba(56,189,248,0.2); height: 8px; border-radius: 4px; margin-top: 10px;">
                        <div style="background: #38bdf8; width: 67%; height: 100%; border-radius: 4px;"></div>
                    </div>
                </div>
            </div>
        `,
        'chart': `
            <div class="component-content">
                <h3>📊 Load History (Last Hour)</h3>
                <div style="height: 200px; background: rgba(0,0,0,0.2); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #71717a;">
                    Chart will render here
                </div>
            </div>
        `,
        'metric': `
            <div class="component-content" style="text-align: center;">
                <div style="color: #71717a; font-size: 0.9rem; margin-bottom: 8px;">RPM</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #38bdf8;">8,234</div>
                <div style="color: #10b981; font-size: 0.85rem; margin-top: 5px;">↑ +234 from setpoint</div>
            </div>
        `,
        'machine-card': `
            <div class="component-content">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3>🔧 CNC_VMC_01</h3>
                    <span style="background: #10b981; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem;">RUNNING</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                    <div>
                        <div style="color: #71717a; font-size: 0.85rem;">Load</div>
                        <div style="font-size: 1.3rem; font-weight: 600;">67%</div>
                    </div>
                    <div>
                        <div style="color: #71717a; font-size: 0.85rem;">RPM</div>
                        <div style="font-size: 1.3rem; font-weight: 600;">8,234</div>
                    </div>
                </div>
            </div>
        `,
        'dopamine': `
            <div class="component-content">
                <h3>🧠 Dopamine Engine</h3>
                <div style="text-align: center; margin: 20px 0;">
                    <div style="font-size: 2.5rem; color: #a855f7;">75%</div>
                    <div style="color: #71717a; margin-top: 5px;">Confidence Level</div>
                </div>
                <div style="background: rgba(168,85,247,0.1); padding: 12px; border-radius: 8px; font-size: 0.9rem;">
                    💭 "Load stable at 67% for 5min. Vibration low. Rewarding cautious approach."
                </div>
            </div>
        `,
        'signal': `
            <div class="component-content" style="text-align: center;">
                <h3>🚦 Safety Signal</h3>
                <div style="font-size: 5rem; margin: 20px 0;">🟢</div>
                <div style="background: rgba(16,185,129,0.1); padding: 10px; border-radius: 8px;">
                    <div style="color: #10b981; font-weight: 600;">GREEN</div>
                    <div style="color: #71717a; font-size: 0.85rem; margin-top: 4px;">Safe to proceed</div>
                </div>
            </div>
        `,
        'oee': `
            <div class="component-content">
                <h3>🎯 OEE Dashboard</h3>
                <div style="text-align: center; margin: 15px 0;">
                    <div style="font-size: 2.5rem; color: #10b981;">80.6%</div>
                    <div style="color: #71717a;">Overall Equipment Effectiveness</div>
                </div>
                <div style="display: grid; gap: 10px;">
                    <div style="background: rgba(16,185,129,0.1); padding: 10px; border-radius: 6px;">
                        <div style="color: #71717a; font-size: 0.85rem;">Availability</div>
                        <div style="font-weight: 600;">93.8%</div>
                    </div>
                    <div style="background: rgba(56,189,248,0.1); padding: 10px; border-radius: 6px;">
                        <div style="color: #71717a; font-size: 0.85rem;">Performance</div>
                        <div style="font-weight: 600;">90.0%</div>
                    </div>
                    <div style="background: rgba(168,85,247,0.1); padding: 10px; border-radius: 6px;">
                        <div style="color: #71717a; font-size: 0.85rem;">Quality</div>
                        <div style="font-weight: 600;">95.6%</div>
                    </div>
                </div>
            </div>
        `,
        'grid-2col': `
            <div class="component-content" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; min-height: 150px;">
                <div style="background: rgba(255,255,255,0.03); border: 1px dashed #71717a; border-radius: 8px; padding: 20px; display: flex; align-items: center; justify-content: center; color: #71717a;">
                    Column 1
                </div>
                <div style="background: rgba(255,255,255,0.03); border: 1px dashed #71717a; border-radius: 8px; padding: 20px; display: flex; align-items: center; justify-content: center; color: #71717a;">
                    Column 2
                </div>
            </div>
        `,
        'card': `
            <div class="component-content" style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 24px; min-height: 150px;">
                <h3 style="margin-bottom: 15px;">Card Title</h3>
                <p style="color: #71717a;">Card content goes here...</p>
            </div>
        `
    };

    return templates[type] || '<div>Unknown component</div>';
}

// === COMPONENT SELECTION ===

function selectComponent(componentId) {
    // Deselect all
    document.querySelectorAll('.dropped-component').forEach(el => {
        el.classList.remove('selected');
    });

    // Select clicked
    const component = document.getElementById(componentId);
    if (component) {
        component.classList.add('selected');
        selectedComponent = componentId;
        showProperties(componentId);
    }
}

function showProperties(componentId) {
    const componentData = componentsOnCanvas.find(c => c.id === componentId);
    if (!componentData) return;

    const propertiesContent = document.getElementById('properties-content');
    propertiesContent.innerHTML = `
        <h3>${componentData.type.toUpperCase()}</h3>
        
        <div class="property-group">
            <label class="property-label">Component ID</label>
            <input type="text" class="property-input" value="${componentId}" readonly>
        </div>
        
        <div class="property-group">
            <label class="property-label">Title</label>
            <input type="text" class="property-input" placeholder="Enter title" onchange="updateProperty('${componentId}', 'title', this.value)">
        </div>
        
        <div class="property-group">
            <label class="property-label">Data Source</label>
            <select class="property-input" onchange="updateProperty('${componentId}', 'dataSource', this.value)">
                <option>Select data source...</option>
                <option>Machine Telemetry</option>
                <option>OEE Calculator</option>
                <option>Dopamine Engine</option>
                <option>Economic Data</option>
            </select>
        </div>
        
        <div class="property-group">
            <label class="property-label">Refresh Rate (seconds)</label>
            <input type="number" class="property-input" value="1" min="1" max="60" onchange="updateProperty('${componentId}', 'refreshRate', this.value)">
        </div>
        
        <button class="btn-secondary" style="width: 100%; margin-top: 20px;" onclick="deleteComponent('${componentId}')">
            🗑️ Delete Component
        </button>
    `;
}

// === COMPONENT ACTIONS ===

function moveUp(componentId) {
    const component = document.getElementById(componentId);
    const prev = component.previousElementSibling;
    if (prev && prev.classList.contains('dropped-component')) {
        component.parentNode.insertBefore(component, prev);
    }
}

function moveDown(componentId) {
    const component = document.getElementById(componentId);
    const next = component.nextElementSibling;
    if (next && next.classList.contains('dropped-component')) {
        component.parentNode.insertBefore(next, component);
    }
}

function duplicateComponent(componentId) {
    const original = componentsOnCanvas.find(c => c.id === componentId);
    if (original) {
        addComponentToCanvas(original.type);
    }
}

function deleteComponent(componentId) {
    const component = document.getElementById(componentId);
    if (component && confirm('Delete this component?')) {
        component.remove();
        componentsOnCanvas = componentsOnCanvas.filter(c => c.id !== componentId);
        document.getElementById('properties-content').innerHTML = '<div class="no-selection"><p>Select a component to edit its properties</p></div>';
    }
}

function updateProperty(componentId, property, value) {
    const component = componentsOnCanvas.find(c => c.id === componentId);
    if (component) {
        component.properties[property] = value;
    }
}

// === CATEGORY TOGGLE ===

function toggleCategory(el) {
    el.parentElement.classList.toggle('collapsed');
}

// === TEMPLATES ===

function loadTemplate(templateName) {
    const canvas = document.getElementById('canvas');
    canvas.innerHTML = ''; // Clear canvas
    componentsOnCanvas = [];

    const templates = {
        'monitoring': ['machine-card', 'telemetry', 'gauge', 'signal'],
        'analytics': ['oee', 'chart', 'metric', 'bar-chart'],
        'cognitive': ['dopamine', 'signal', 'llm-chat', 'chart'],
        'blank': []
    };

    const components = templates[templateName] || [];
    components.forEach(type => {
        addComponentToCanvas(type);
    });

    closeModal();
}

// === ACTIONS ===

function saveDashboard() {
    const dashboardData = {
        components: componentsOnCanvas,
        created: new Date().toISOString()
    };

    localStorage.setItem('dashboard', JSON.stringify(dashboardData));
    alert('✅ Dashboard saved!');
}

function exportCode() {
    const canvas = document.getElementById('canvas');
    const html = canvas.innerHTML;

    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'dashboard.html';
    a.click();

    URL.revokeObjectURL(url);
}

function previewDashboard() {
    const canvas = document.getElementById('canvas');
    const html = canvas.innerHTML;

    const previewWindow = window.open('', '_blank');
    previewWindow.document.write(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard Preview</title>
            <link rel="stylesheet" href="dashboard-builder.css">
        </head>
        <body style="padding: 20px;">
            ${html}
        </body>
        </html>
    `);
}

function closeModal() {
    document.getElementById('templates-modal').style.display = 'none';
}

function getDefaultProperties(componentType) {
    return {
        title: '',
        dataSource: '',
        refreshRate: 1
    };
}

function undoAction() {
    alert('Undo not yet implemented');
}

function redoAction() {
    alert('Redo not yet implemented');
}
