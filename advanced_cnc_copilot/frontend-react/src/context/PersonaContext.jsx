import React, { createContext, useContext, useState, useMemo } from 'react';
import tokens from '../design-tokens.json';

const PersonaContext = createContext();

export const usePersona = () => useContext(PersonaContext);

export const PersonaProvider = ({ children }) => {
    // Site configuration mapped to persona roles
    const siteConfigs = {
        OPERATOR: {
            primary: tokens.theme.colors.industrial.primary, // #FFD700 (Adjusted to tokens, user suggested Safety Orange #FF5F1F)
            accent: tokens.theme.colors.neuro.success,
            theme: 'industrial',
            borderRadius: '0px',
            title: 'EXECUTION HUD'
        },
        MANAGER: {
            primary: tokens.theme.colors.neuro.success, // Emerald Logic
            accent: tokens.theme.colors.neuro.synapse,
            theme: 'managerial',
            borderRadius: '8px',
            title: 'FLEET COMMAND'
        },
        ENGINEER: {
            primary: tokens.theme.colors.neuro.pulse, // Deep Cyber-Blue base
            accent: tokens.theme.colors.neuro.synapse,
            theme: 'creative',
            borderRadius: '12px',
            title: 'GENERATIVE STUDIO'
        },
        ADMIN: {
            primary: tokens.theme.colors.neuro.danger, // Neural Purple logic
            accent: tokens.theme.colors.neuro.pulse,
            theme: 'root',
            borderRadius: '4px',
            title: 'ROOT CONSOLE'
        }
    };

    const [activePersona, setActivePersona] = useState('OPERATOR');

    const config = useMemo(() => siteConfigs[activePersona] || siteConfigs.OPERATOR, [activePersona]);

    // Inject CSS variables for dynamic tokens
    React.useEffect(() => {
        const root = document.documentElement;
        root.style.setProperty('--persona-primary', config.primary);
        root.style.setProperty('--persona-accent', config.accent);
        root.style.setProperty('--persona-radius', config.borderRadius);
    }, [config]);

    return (
        <PersonaContext.Provider value={{ persona: activePersona, setPersona: setActivePersona, config }}>
            {children}
        </PersonaContext.Provider>
    );
};
