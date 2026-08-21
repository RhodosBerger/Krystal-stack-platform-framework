/**
 * RISE Setup Wizard Logic
 * Handles state, form rendering, and AI interaction
 */

let currentStep = 0;
let wizardConfig = null;
let formData = {};

// 1. Initial Load
async function initWizard() {
    try {
        const response = await fetch('/api/wizard/config');
        wizardConfig = await response.json();
        renderSteps();
        showStep(0);
    } catch (error) {
        console.error("Failed to load wizard config:", error);
        alert("Critical Error: Could not load configuration. Ensure backend is running.");
    }
}

// 2. Render Step Forms
function renderSteps() {
    const container = document.getElementById('step-container');
    container.innerHTML = '';

    wizardConfig.steps.forEach((step, index) => {
        const stepDiv = document.createElement('div');
        stepDiv.className = `step-content ${index === 0 ? 'active' : ''}`;
        stepDiv.id = `step-${index}`;

        let fieldsHtml = '';
        step.fields.forEach(field => {
            fieldsHtml += `
                <div class="form-group">
                    <label for="${field.id}">${field.label}</label>
                    ${renderField(field)}
                </div>
            `;
        });

        stepDiv.innerHTML = `
            <div class="step-header">
                <h1>${step.icon} ${step.title}</h1>
                <p>${step.description}</p>
            </div>
            <form id="form-${index}">
                ${fieldsHtml}
            </form>
        `;
        container.appendChild(stepDiv);
    });
}

function renderField(field) {
    const value = field.default || '';

    if (field.type === 'text') {
        return `<input type="text" id="${field.id}" placeholder="${field.placeholder || ''}" value="${value}">`;
    }
    if (field.type === 'select') {
        let options = '';
        field.options.forEach(opt => {
            options += `<option value="${opt.value}" ${opt.value === value ? 'selected' : ''}>${opt.label}</option>`;
        });
        return `<select id="${field.id}">${options}</select>`;
    }
    if (field.type === 'toggle') {
        return `
            <div class="toggle-container">
                <span style="color:#888; font-size:0.9rem;">Enable/Disable</span>
                <input type="checkbox" id="${field.id}" ${value ? 'checked' : ''} style="width:20px; height:20px; cursor:pointer;">
            </div>
        `;
    }
    if (field.type === 'slider') {
        return `
            <div style="display:flex; align-items:center; gap:15px;">
                <input type="range" id="${field.id}" min="${field.min}" max="${field.max}" value="${value}" style="flex-grow:1; accent-color:var(--primary);">
                <span id="val-${field.id}" style="font-family:monospace; color:var(--primary); font-weight:bold;">${value}</span>
            </div>
            <script>
                document.getElementById('${field.id}').oninput = function() {
                    document.getElementById('val-${field.id}').innerText = this.value;
                }
            </script>
        `;
    }
    return `<span>Unsupported Field: ${field.type}</span>`;
}

// 3. Navigation
function showStep(index) {
    // Update step visibility
    document.querySelectorAll('.step-content').forEach((el, i) => {
        el.classList.toggle('active', i === index);
    });

    // Update sidebar items
    document.querySelectorAll('.step-item').forEach((el, i) => {
        el.classList.toggle('active', i === index);
        if (i < index) el.classList.add('completed');
    });

    // Update buttons
    document.getElementById('btn-prev').style.visibility = index === 0 ? 'hidden' : 'visible';
    const nextBtn = document.getElementById('btn-next');

    if (index === wizardConfig.steps.length - 1) {
        nextBtn.innerText = 'Seal Configuration';
        nextBtn.classList.remove('btn-next');
        nextBtn.classList.add('btn-finish');
        nextBtn.onclick = finalizeWizard;
    } else {
        nextBtn.innerText = 'Next';
        nextBtn.classList.add('btn-next');
        nextBtn.classList.remove('btn-finish');
        nextBtn.onclick = () => moveStep(1);
    }

    currentStep = index;
}

function moveStep(direction) {
    const nextStep = currentStep + direction;
    if (nextStep >= 0 && nextStep < wizardConfig.steps.length) {
        // Collect current step data
        const currentForm = wizardConfig.steps[currentStep];
        currentForm.fields.forEach(field => {
            const el = document.getElementById(field.id);
            if (el) {
                formData[field.id] = field.type === 'toggle' ? el.checked : el.value;
            }
        });
        showStep(nextStep);
    }
}

async function finalizeWizard() {
    // Add final data
    const lastForm = wizardConfig.steps[currentStep];
    lastForm.fields.forEach(field => {
        const el = document.getElementById(field.id);
        if (el) formData[field.id] = field.type === 'toggle' ? el.checked : el.value;
    });

    console.log("Finalized Data:", formData);

    // UI Feedback
    const container = document.getElementById('step-container');
    container.innerHTML = `
        <div class="step-header" style="text-align:center; padding-top:50px;">
            <div style="font-size:5rem; color:var(--primary); margin-bottom:20px;">🛡️</div>
            <h1>SYSTEM SEALED</h1>
            <p>Your configuration is stored. The CNC Copilot is active.</p>
            <button class="nav-btn btn-next" onclick="window.location.href='/view/neuro'" style="margin-top:20px;">Enter Command Center</button>
        </div>
    `;
    document.querySelector('.nav-buttons').style.display = 'none';
}

// 4. AI Assistant Interaction
function toggleAI() {
    const chat = document.getElementById('ai-chat');
    chat.classList.toggle('open');
}

async function aiAsk() {
    const input = document.getElementById('ai-input');
    const window = document.getElementById('chat-window');
    const question = input.value.trim();

    if (!question) return;

    // Add user message
    const userMsg = document.createElement('div');
    userMsg.style.cssText = "background:rgba(255,255,255,0.1); padding:8px; border-radius:8px; margin-bottom:10px; text-align:right;";
    userMsg.innerText = question;
    window.appendChild(userMsg);
    input.value = '';

    // Add loading
    const loadingMsg = document.createElement('div');
    loadingMsg.innerText = "Thinking...";
    window.appendChild(loadingMsg);
    window.scrollTop = window.scrollHeight;

    try {
        const response = await fetch('/api/intelligence/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: `In the context of the Setup Wizard: ${question}` })
        });
        const data = await response.json();

        loadingMsg.innerHTML = `
            <div style="background:var(--glass); padding:10px; border-radius:8px; margin-bottom:10px; border-left: 2px solid var(--primary);">
                ${data.answer}
            </div>
        `;
    } catch (error) {
        loadingMsg.innerText = "Error: Could not connect to AI backend.";
    }
    window.scrollTop = window.scrollHeight;
}

function handleAIChat(event) {
    if (event.key === 'Enter') aiAsk();
}

// Start
initWizard();
