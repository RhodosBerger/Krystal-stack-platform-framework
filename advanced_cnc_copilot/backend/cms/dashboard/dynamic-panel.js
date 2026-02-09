// ====================
// DYNAMIC PANEL - FRONTEND
// Backend → Visual Representation
// ====================

class DynamicPanelRenderer {
    constructor() {
        this.config = null;
        this.values = {};
        this.formContainer = document.getElementById('dynamic-form');
        this.apiBase = 'http://localhost:8000/api';
    }

    /**
     * STEP 1: Načítaj config z backendu
     */
    async loadConfig(panelType = 'safety') {
        try {
            const response = await fetch(`${this.apiBase}/config/${panelType}`);
            this.config = await response.json();
            this.renderForm();
        } catch (error) {
            console.error('Failed to load config:', error);
            this.showError('Nepodarilo sa načítať konfiguráciu');
        }
    }

    /**
     * STEP 2: Vygeneruj HTML zo sekci
     */
    renderForm() {
        if (!this.config || !this.config.sections) return;

        this.formContainer.innerHTML = '';

        this.config.sections.forEach(section => {
            const sectionEl = this.createSection(section);
            this.formContainer.appendChild(sectionEl);
        });

        // Initialize all field values
        this.initializeValues();

        // Setup conditional logic watchers
        this.setupConditionalLogic();
    }

    /**
     * Vytvor HTML pre sekciu
     */
    createSection(section) {
        const sectionDiv = document.createElement('div');
        sectionDiv.className = 'form-section';
        sectionDiv.id = `section-${section.id}`;

        // Section header
        const header = document.createElement('h3');
        header.innerHTML = `${section.icon} ${section.title}`;
        sectionDiv.appendChild(header);

        if (section.description) {
            const desc = document.createElement('p');
            desc.className = 'section-description';
            desc.textContent = section.description;
            sectionDiv.appendChild(desc);
        }

        // Render all fields
        section.fields.forEach(field => {
            const fieldEl = this.createField(field);
            sectionDiv.appendChild(fieldEl);
        });

        return sectionDiv;
    }

    /**
     * STEP 3: Vytvor HTML pre field podľa typu
     */
    createField(field) {
        const fieldDiv = document.createElement('div');
        fieldDiv.className = 'form-field';
        fieldDiv.id = `field-${field.id}`;
        fieldDiv.dataset.fieldId = field.id;

        // Label
        const label = document.createElement('label');
        label.htmlFor = field.id;
        label.innerHTML = field.label;
        if (field.required) {
            label.innerHTML += ' <span class="required">*</span>';
        }
        fieldDiv.appendChild(label);

        // Field-specific rendering
        let inputEl;
        switch (field.type) {
            case 'text':
            case 'email':
            case 'number':
                inputEl = this.createTextInput(field);
                break;
            case 'slider':
                inputEl = this.createSlider(field);
                break;
            case 'toggle':
                inputEl = this.createToggle(field);
                break;
            case 'select':
                inputEl = this.createSelect(field);
                break;
            case 'multiselect':
                inputEl = this.createMultiSelect(field);
                break;
            case 'tags':
                inputEl = this.createTagsInput(field);
                break;
            case 'textarea':
                inputEl = this.createTextarea(field);
                break;
            default:
                inputEl = this.createTextInput(field);
        }

        fieldDiv.appendChild(inputEl);

        // Help text
        if (field.help) {
            const helpText = document.createElement('small');
            helpText.className = 'help-text';
            helpText.textContent = field.help;
            fieldDiv.appendChild(helpText);
        }

        // Validation error placeholder
        const errorDiv = document.createElement('div');
        errorDiv.className = 'field-error';
        errorDiv.style.display = 'none';
        fieldDiv.appendChild(errorDiv);

        return fieldDiv;
    }

    /**
     * TEXT INPUT (text, email, number)
     */
    createTextInput(field) {
        const input = document.createElement('input');
        input.type = field.type;
        input.id = field.id;
        input.name = field.id;
        input.value = field.default || '';
        input.placeholder = field.placeholder || '';

        if (field.required) input.required = true;
        if (field.min !== null) input.min = field.min;
        if (field.max !== null) input.max = field.max;
        if (field.pattern) input.pattern = field.pattern;

        input.addEventListener('change', (e) => {
            this.values[field.id] = e.target.value;
            this.handleFieldChange(field.id);
        });

        return input;
    }

    /**
     * SLIDER
     */
    createSlider(field) {
        const container = document.createElement('div');
        container.className = 'slider-container';

        const valueDisplay = document.createElement('div');
        valueDisplay.className = 'slider-value';
        valueDisplay.textContent = field.default || field.min || 0;
        container.appendChild(valueDisplay);

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.className = 'slider';
        slider.id = field.id;
        slider.name = field.id;
        slider.min = field.min || 0;
        slider.max = field.max || 100;
        slider.step = field.step || 1;
        slider.value = field.default || field.min || 0;

        // Update gradient based on value
        const updateGradient = () => {
            const percent = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
            slider.style.setProperty('--value', `${percent}%`);
        };
        updateGradient();

        slider.addEventListener('input', (e) => {
            valueDisplay.textContent = e.target.value;
            updateGradient();
        });

        slider.addEventListener('change', (e) => {
            this.values[field.id] = parseFloat(e.target.value);
            this.handleFieldChange(field.id);
        });

        container.appendChild(slider);
        return container;
    }

    /**
     * TOGGLE SWITCH
     */
    createToggle(field) {
        const label = document.createElement('label');
        label.className = 'toggle-switch';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = field.id;
        input.name = field.id;
        input.checked = field.default || false;

        const span = document.createElement('span');
        span.className = 'toggle-slider';

        label.appendChild(input);
        label.appendChild(span);

        input.addEventListener('change', (e) => {
            this.values[field.id] = e.target.checked;
            this.handleFieldChange(field.id);
        });

        return label;
    }

    /**
     * SELECT DROPDOWN
     */
    createSelect(field) {
        const select = document.createElement('select');
        select.id = field.id;
        select.name = field.id;

        field.options.forEach(option => {
            const optionEl = document.createElement('option');
            optionEl.value = option.value;
            optionEl.textContent = option.label;
            if (option.value === field.default) {
                optionEl.selected = true;
            }
            select.appendChild(optionEl);
        });

        select.addEventListener('change', (e) => {
            this.values[field.id] = e.target.value;
            this.handleFieldChange(field.id);
        });

        return select;
    }

    /**
     * MULTISELECT (checkbox list)
     */
    createMultiSelect(field) {
        const container = document.createElement('div');
        container.className = 'multiselect-container';

        const defaults = field.default || [];

        field.options.forEach(option => {
            const label = document.createElement('label');
            label.className = 'checkbox-label';

            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.value = option.value;
            checkbox.checked = defaults.includes(option.value);

            checkbox.addEventListener('change', () => {
                const checked = container.querySelectorAll('input[type="checkbox"]:checked');
                this.values[field.id] = Array.from(checked).map(cb => cb.value);
                this.handleFieldChange(field.id);
            });

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(' ' + option.label));
            container.appendChild(label);
        });

        return container;
    }

    /**
     * TAGS INPUT
     */
    createTagsInput(field) {
        const container = document.createElement('div');
        container.className = 'tag-input';
        container.id = `tags-${field.id}`;

        const tags = field.default || [];
        this.values[field.id] = [...tags];

        // Render existing tags
        tags.forEach(tag => {
            const tagEl = this.createTag(tag, field.id);
            container.appendChild(tagEl);
        });

        // Input for new tag
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = field.placeholder || 'Add tag...';
        input.style.border = 'none';
        input.style.background = 'transparent';
        input.style.outline = 'none';
        input.style.flex = '1';

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && input.value.trim()) {
                e.preventDefault();
                const newTag = input.value.trim();

                if (!this.values[field.id].includes(newTag)) {
                    this.values[field.id].push(newTag);
                    const tagEl = this.createTag(newTag, field.id);
                    container.insertBefore(tagEl, input);
                    this.handleFieldChange(field.id);
                }

                input.value = '';
            }
        });

        container.appendChild(input);
        return container;
    }

    createTag(text, fieldId) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.textContent = text;

        const remove = document.createElement('span');
        remove.className = 'remove';
        remove.textContent = '×';
        remove.onclick = () => {
            tag.remove();
            this.values[fieldId] = this.values[fieldId].filter(t => t !== text);
            this.handleFieldChange(fieldId);
        };

        tag.appendChild(remove);
        return tag;
    }

    /**
     * TEXTAREA
     */
    createTextarea(field) {
        const textarea = document.createElement('textarea');
        textarea.id = field.id;
        textarea.name = field.id;
        textarea.value = field.default || '';
        textarea.placeholder = field.placeholder || '';
        textarea.rows = 4;

        textarea.addEventListener('change', (e) => {
            this.values[field.id] = e.target.value;
            this.handleFieldChange(field.id);
        });

        return textarea;
    }

    /**
     * STEP 4: Initialize values from defaults
     */
    initializeValues() {
        this.config.sections.forEach(section => {
            section.fields.forEach(field => {
                if (field.default !== null && field.default !== undefined) {
                    this.values[field.id] = field.default;
                }
            });
        });
    }

    /**
     * STEP 5: Setup conditional logic (depends_on, show_if)
     */
    setupConditionalLogic() {
        this.config.sections.forEach(section => {
            section.fields.forEach(field => {
                if (field.depends_on) {
                    this.updateFieldVisibility(field);
                }
            });
        });
    }

    /**
     * Handle field value change
     */
    handleFieldChange(fieldId) {
        console.log(`Field ${fieldId} changed to:`, this.values[fieldId]);

        // Update fields that depend on this one
        this.config.sections.forEach(section => {
            section.fields.forEach(field => {
                if (field.depends_on === fieldId) {
                    this.updateFieldVisibility(field);
                }
            });
        });

        // Trigger auto-save (if enabled)
        this.autoSave();
    }

    /**
     * Update field visibility based on condition
     */
    updateFieldVisibility(field) {
        const fieldEl = document.getElementById(`field-${field.id}`);
        if (!fieldEl) return;

        const dependsOnValue = this.values[field.depends_on];
        const shouldShow = dependsOnValue === field.show_if;

        fieldEl.style.display = shouldShow ? 'block' : 'none';
    }

    /**
     * STEP 6: Validation
     */
    validate() {
        let isValid = true;
        const errors = {};

        this.config.sections.forEach(section => {
            section.fields.forEach(field => {
                const fieldEl = document.getElementById(`field-${field.id}`);
                if (!fieldEl || fieldEl.style.display === 'none') return;

                const value = this.values[field.id];
                const fieldErrors = [];

                // Required check
                if (field.required && (value === null || value === undefined || value === '')) {
                    fieldErrors.push(`${field.label} je povinný`);
                }

                // Type-specific validation
                if (value !== null && value !== undefined) {
                    if (field.type === 'number' || field.type === 'slider') {
                        if (field.min !== null && value < field.min) {
                            fieldErrors.push(`Minimum je ${field.min}`);
                        }
                        if (field.max !== null && value > field.max) {
                            fieldErrors.push(`Maximum je ${field.max}`);
                        }
                    }

                    if (field.pattern && typeof value === 'string') {
                        const regex = new RegExp(field.pattern);
                        if (!regex.test(value)) {
                            fieldErrors.push('Neplatný formát');
                        }
                    }
                }

                if (fieldErrors.length > 0) {
                    isValid = false;
                    errors[field.id] = fieldErrors;
                    this.showFieldError(field.id, fieldErrors);
                } else {
                    this.hideFieldError(field.id);
                }
            });
        });

        return { isValid, errors };
    }

    showFieldError(fieldId, errors) {
        const fieldEl = document.getElementById(`field-${fieldId}`);
        const errorDiv = fieldEl.querySelector('.field-error');
        errorDiv.textContent = errors.join(', ');
        errorDiv.style.display = 'block';
        errorDiv.style.color = '#ef4444';
        errorDiv.style.fontSize = '0.85rem';
        errorDiv.style.marginTop = '4px';
    }

    hideFieldError(fieldId) {
        const fieldEl = document.getElementById(`field-${fieldId}`);
        const errorDiv = fieldEl.querySelector('.field-error');
        errorDiv.style.display = 'none';
    }

    /**
     * STEP 7: Submit
     */
    async submit() {
        const validation = this.validate();

        if (!validation.isValid) {
            alert('Opravte chyby vo formulári');
            return false;
        }

        try {
            const response = await fetch(`${this.apiBase}/config/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.values)
            });

            if (response.ok) {
                this.showSuccess('Konfigurácia uložená!');
                return true;
            } else {
                this.showError('Nepodarilo sa uložiť');
                return false;
            }
        } catch (error) {
            console.error('Submit failed:', error);
            this.showError('Chyba pripojenia');
            return false;
        }
    }

    /**
     * Auto-save (debounced)
     */
    autoSave() {
        clearTimeout(this.autoSaveTimeout);
        this.autoSaveTimeout = setTimeout(() => {
            console.log('Auto-saving...', this.values);
            // Silent save without validation
        }, 2000);
    }

    /**
     * UI Helpers
     */
    showError(message) {
        // TODO: Implement toast notification
        alert('Error: ' + message);
    }

    showSuccess(message) {
        // TODO: Implement toast notification
        alert('Success: ' + message);
    }
}

// ====================
// NAVIGATION LOGIC
// ====================

function toggleNav(headerEl) {
    const navItem = headerEl.parentElement;
    navItem.classList.toggle('expanded');
}

function loadPanel(feature) {
    console.log('Loading panel:', feature);

    // Update active state
    document.querySelectorAll('.nav-header').forEach(h => h.classList.remove('active'));
    event.target.closest('.nav-header').classList.add('active');

    // Update panel title
    document.getElementById('panel-title').textContent = feature.toUpperCase();

    // Load config for this panel
    window.panelRenderer.loadConfig(feature);
}

function toggleSidebar() {
    const sidebar = document.getElementById('main-sidebar');
    sidebar.classList.toggle('collapsed');
    document.body.classList.toggle('sidebar-collapsed');
}

function collapseSidebar() {
    const sidebar = document.getElementById('main-sidebar');
    sidebar.classList.add('collapsed');
    document.body.classList.add('sidebar-collapsed');
}

// ====================
// FORM ACTIONS
// ====================

function saveConfiguration() {
    window.panelRenderer.submit();
}

function resetToDefaults() {
    if (confirm('Resetovať na predvolené hodnoty?')) {
        window.panelRenderer.initializeValues();
        window.panelRenderer.renderForm();
    }
}

function exportConfig() {
    const dataStr = JSON.stringify(window.panelRenderer.values, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'fanuc-rise-config.json';
    a.click();

    URL.revokeObjectURL(url);
}

// ====================
// INITIALIZATION
// ====================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize renderer
    window.panelRenderer = new DynamicPanelRenderer();

    // Load default panel (safety)
    window.panelRenderer.loadConfig('safety');

    console.log('✅ Dynamic Panel initialized');
});
