import React, { useState } from 'react';
import { 
  Plug, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  RefreshCw, 
  ExternalLink,
  Key,
  Database,
  Globe,
  Cpu
} from 'lucide-react';

const IntegrationCard = ({ title, description, status, icon: Icon, type, connected }) => {
  return (
    <div className="bg-industrial-surface border border-white/5 rounded-xl p-6 relative overflow-hidden group hover:border-white/10 transition-all">
      <div className="flex justify-between items-start mb-4">
        <div className="flex items-center gap-4">
          <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${connected ? 'bg-neuro-success/10 text-neuro-success' : 'bg-white/5 text-gray-400'}`}>
            <Icon size={24} />
          </div>
          <div>
            <h3 className="font-bold text-white text-lg">{title}</h3>
            <span className="text-[10px] font-mono text-gray-500 uppercase tracking-wider">{type}</span>
          </div>
        </div>
        <div className={`px-2 py-1 rounded text-[10px] font-bold font-mono border ${
          connected 
            ? 'bg-neuro-success/10 text-neuro-success border-neuro-success/20' 
            : 'bg-white/5 text-gray-500 border-white/10'
        }`}>
          {connected ? 'CONNECTED' : 'DISCONNECTED'}
        </div>
      </div>
      
      <p className="text-sm text-gray-400 mb-6 leading-relaxed">
        {description}
      </p>

      <div className="flex items-center gap-3">
        <button className={`flex-1 py-2 rounded-lg text-xs font-bold font-mono transition-all flex items-center justify-center gap-2 ${
          connected 
            ? 'bg-white/5 text-white hover:bg-white/10' 
            : 'bg-industrial-primary text-black hover:bg-industrial-primary/90'
        }`}>
          {connected ? <SettingsIcon /> : <Plug size={14} />}
          {connected ? 'CONFIGURE' : 'CONNECT'}
        </button>
        {connected && (
          <button className="p-2 rounded-lg bg-white/5 text-gray-400 hover:text-white hover:bg-white/10">
            <RefreshCw size={14} />
          </button>
        )}
      </div>

      {/* Decorative Status Line */}
      <div className={`absolute bottom-0 left-0 w-full h-1 ${connected ? 'bg-neuro-success' : 'bg-gray-800'}`} />
    </div>
  );
};

const SettingsIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.1a2 2 0 0 1-1-1.72v-.51a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"></path>
    <circle cx="12" cy="12" r="3"></circle>
  </svg>
);

const IntegrationsPanel = () => {
  const integrations = [
    {
      id: 'openai',
      title: 'OpenAI GPT-4',
      type: 'Intelligence',
      description: 'Primary cognitive engine for G-Code generation, semantic analysis, and natural language processing.',
      icon: Cpu,
      connected: true,
      status: 'operational'
    },
    {
      id: 'fanuc',
      title: 'Fanuc FOCAS',
      type: 'Hardware Bridge',
      description: 'Direct Ethernet connection to Fanuc CNC controllers via FOCAS2 library protocol.',
      icon: Database,
      connected: true,
      status: 'operational'
    },
    {
      id: 'aws',
      title: 'AWS IoT Core',
      type: 'Cloud',
      description: 'Telemetry streaming and long-term storage for digital twin synchronization.',
      icon: Globe,
      connected: false,
      status: 'disconnected'
    },
    {
      id: 'slack',
      title: 'Slack Alerts',
      type: 'Notification',
      description: 'Real-time critical alerts and shift reports sent to engineering channels.',
      icon: AlertTriangle,
      connected: false,
      status: 'disconnected'
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-white mb-1">Integrations</h2>
          <p className="text-gray-400 text-sm font-mono">Manage external connections and API gateways.</p>
        </div>
        <button className="px-4 py-2 bg-white/5 border border-white/10 rounded-lg text-xs font-bold text-white hover:bg-white/10 transition-colors flex items-center gap-2">
            <RefreshCw size={14} /> CHECK STATUS
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {integrations.map(integ => (
          <IntegrationCard key={integ.id} {...integ} />
        ))}
      </div>

      {/* API Key Management Section */}
      <div className="bg-industrial-surface border border-white/5 rounded-xl p-6 mt-8">
        <h3 className="font-bold text-white mb-6 flex items-center gap-2">
            <Key size={18} className="text-industrial-primary" /> 
            Secure Keys Vault
        </h3>
        
        <div className="space-y-4">
            {['OPENAI_API_KEY', 'FANUC_LICENSE_KEY', 'AWS_ACCESS_SECRET'].map((key, i) => (
                <div key={key} className="flex items-center gap-4 p-4 bg-black/20 rounded-lg border border-white/5">
                    <div className="w-8 h-8 rounded bg-white/5 flex items-center justify-center text-gray-500">
                        <Key size={14} />
                    </div>
                    <div className="flex-1">
                        <div className="text-[10px] font-mono text-gray-500 mb-1">KEY_ID</div>
                        <div className="text-sm font-mono text-white font-bold">{key}</div>
                    </div>
                    <div className="font-mono text-xs text-gray-500 tracking-widest">
                        ••••••••••••••••x8G2
                    </div>
                    <button className="px-3 py-1 text-[10px] font-bold border border-white/10 rounded text-gray-400 hover:text-white hover:border-white/30">
                        ROTATE
                    </button>
                </div>
            ))}
        </div>
      </div>
    </div>
  );
};

export default IntegrationsPanel;
