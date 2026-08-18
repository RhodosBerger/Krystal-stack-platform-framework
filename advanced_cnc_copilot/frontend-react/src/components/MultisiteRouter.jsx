import React, { useState, useEffect } from 'react';
import OperatorLayout from '../layouts/OperatorLayout';
import ManagerLayout from '../layouts/ManagerLayout';
import CreativeLayout from '../layouts/CreativeLayout';
import ConfigurationLayout from '../layouts/ConfigurationLayout';
import ResourceLayout from '../layouts/ResourceLayout';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { Lock } from 'lucide-react';

import { usePersona } from '../context/PersonaContext';

const MultisiteRouter = () => {
  const { user, login, hasPermission } = useAuth();
  const { persona, setPersona } = usePersona();

  const handlePersonaShift = (newRole) => {
    login(newRole);
    setPersona(newRole);
  };

  const roles = [
    { id: 'OPERATOR', label: 'OPERATOR', color: 'bg-[#FF5F1F]', required: 'OPERATOR' },
    { id: 'MANAGER', label: 'MANAGER', color: 'bg-neuro-success', required: 'MANAGER' },
    { id: 'ENGINEER', label: 'CREATOR', color: 'bg-neuro-pulse', required: 'ENGINEER' },
    { id: 'ADMIN', label: 'ADMIN', color: 'bg-neuro-danger', required: 'ADMIN' }
  ];

  return (
    <>
      {/* macOS Style Dock (Persona Switcher) */}
      <div className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50">
        <div className="bg-white/80 backdrop-blur-2xl px-4 py-3 rounded-2xl shadow-[0_8px_40px_rgba(0,0,0,0.12)] border border-white/40 flex items-center gap-4 transition-all hover:scale-[1.02]">
          {roles.map(role => (
            <button
              key={role.id}
              onClick={() => handlePersonaShift(role.id)}
              className={`relative group flex flex-col items-center gap-1 transition-all duration-300 ${user.role === role.id ? '-translate-y-2' : 'hover:-translate-y-1'}`}
            >
              <div
                className={`w-12 h-12 rounded-2xl flex items-center justify-center shadow-md transition-all ${user.role === role.id
                    ? 'bg-gradient-to-br from-gray-800 to-black text-white shadow-xl scale-110'
                    : 'bg-white text-gray-500 hover:bg-gray-50'
                  }`}
              >
                {/* Icons mapped to roles */}
                {role.id === 'OPERATOR' && <span className="font-bold">OP</span>}
                {role.id === 'MANAGER' && <span className="font-bold">MG</span>}
                {role.id === 'ENGINEER' && <span className="font-bold">CR</span>}
                {role.id === 'ADMIN' && <span className="font-bold">AD</span>}
              </div>
              <span className={`text-[10px] font-medium tracking-wide transition-opacity ${user.role === role.id ? 'opacity-100 text-black' : 'opacity-0 group-hover:opacity-60'}`}>
                {role.label}
              </span>

              {/* Active Indicator Dot */}
              {user.role === role.id && (
                <div className="absolute -bottom-5 w-1 h-1 rounded-full bg-black/30" />
              )}
            </button>
          ))}
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={persona}
          initial={{ opacity: 0, filter: 'blur(20px)' }}
          animate={{ opacity: 1, filter: 'blur(0px)' }}
          exit={{ opacity: 0, filter: 'blur(20px)' }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="min-h-screen"
        >
          {persona === 'OPERATOR' && <OperatorLayout />}
          {persona === 'MANAGER' && <ManagerLayout />}
          {persona === 'ENGINEER' && <CreativeLayout />}
          {persona === 'ADMIN' && <ConfigurationLayout />}
        </motion.div>
      </AnimatePresence>
    </>
  );
};

export default MultisiteRouter;
