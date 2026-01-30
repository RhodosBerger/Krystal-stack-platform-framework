import React, { useState } from 'react';
import { Box, Layers, Droplet, Hexagon } from 'lucide-react';

const MaterialSelector = () => {
    const [selected, setSelected] = useState('TITANIUM_6AL_4V');

    const materials = [
        { id: 'TITANIUM_6AL_4V', name: 'Titanium Grade 5', type: 'Metal', density: '4.43 g/cm³', hardness: '36 HRC', icon: <Hexagon size={24} /> },
        { id: 'ALUMINUM_6061', name: 'Aluminum 6061-T6', type: 'Metal', density: '2.70 g/cm³', hardness: '95 HB', icon: <Box size={24} /> },
        { id: 'PEEK_MEDICAL', name: 'PEEK (Medical)', type: 'Polymer', density: '1.32 g/cm³', hardness: '85 Shore D', icon: <Layers size={24} /> },
        { id: 'INCONEL_718', name: 'Inconel 718', type: 'Superalloy', density: '8.19 g/cm³', hardness: '40 HRC', icon: <Droplet size={24} /> },
    ];

    return (
        <div className="grid grid-cols-1 gap-3">
            {materials.map((mat) => (
                <div 
                    key={mat.id}
                    onClick={() => setSelected(mat.id)}
                    className={`group relative p-4 rounded-lg border cursor-pointer transition-all duration-300 overflow-hidden ${
                        selected === mat.id 
                        ? 'bg-neuro-pulse/10 border-neuro-pulse' 
                        : 'bg-black/20 border-white/5 hover:border-white/20'
                    }`}
                >
                    {/* Active Indicator */}
                    {selected === mat.id && (
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-neuro-pulse shadow-[0_0_10px_#00FFC8]" />
                    )}

                    <div className="flex items-center gap-4">
                        <div className={`p-3 rounded-md transition-colors ${
                            selected === mat.id ? 'bg-neuro-pulse text-black' : 'bg-white/5 text-gray-400 group-hover:text-white'
                        }`}>
                            {mat.icon}
                        </div>
                        <div className="flex-1">
                            <h4 className={`text-xs font-bold tracking-wider mb-1 ${selected === mat.id ? 'text-white' : 'text-gray-300'}`}>
                                {mat.name}
                            </h4>
                            <div className="flex gap-3 text-[10px] neuro-text text-gray-500">
                                <span>{mat.type}</span>
                                <span>•</span>
                                <span>{mat.density}</span>
                            </div>
                        </div>
                        {selected === mat.id && (
                            <div className="text-[9px] font-mono text-neuro-pulse animate-pulse">ACTIVE</div>
                        )}
                    </div>
                </div>
            ))}
        </div>
    );
};

export default MaterialSelector;