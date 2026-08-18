import React, { useState } from 'react';
import AppleCard from './AppleCard';
import {
    LayoutDashboard,
    BarChart3,
    Activity,
    Zap,
    Gauge,
    Type,
    Settings,
    Trash2,
    MoveUp,
    MoveDown,
    Copy
} from 'lucide-react';

const COMPONENT_TYPES = [
    { type: 'gauge', label: 'Metric Gauge', icon: Gauge, defaultProps: { title: 'Spindle Load', value: '67%', unit: '%' } },
    { type: 'chart', label: 'History Chart', icon: BarChart3, defaultProps: { title: 'Load History', height: '200px' } },
    { type: 'stat', label: 'Single Stat', icon: Activity, defaultProps: { title: 'RPM', value: '8,234', trend: '+234' } },
    { type: 'text', label: 'Text Block', icon: Type, defaultProps: { content: 'Add notes or instructions here...' } },
    { type: 'control', label: 'Control Switch', icon: Zap, defaultProps: { label: 'Coolant Pump' } },
];

const DashboardBuilder = () => {
    const [components, setComponents] = useState([]);
    const [selectedId, setSelectedId] = useState(null);
    const [draggedType, setDraggedType] = useState(null);

    const handleDragStart = (e, type) => {
        setDraggedType(type);
        e.dataTransfer.effectAllowed = 'copy';
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (draggedType) {
            const typeDef = COMPONENT_TYPES.find(t => t.type === draggedType);
            const newComponent = {
                id: Date.now().toString(),
                type: draggedType,
                props: { ...typeDef.defaultProps }
            };
            setComponents([...components, newComponent]);
            setSelectedId(newComponent.id);
            setDraggedType(null);
        }
    };

    const updateProperty = (key, value) => {
        setComponents(components.map(c =>
            c.id === selectedId ? { ...c, props: { ...c.props, [key]: value } } : c
        ));
    };

    const deleteComponent = (id) => {
        setComponents(components.filter(c => c.id !== id));
        if (selectedId === id) setSelectedId(null);
    };

    const moveComponent = (index, direction) => {
        const newComponents = [...components];
        const [moved] = newComponents.splice(index, 1);
        newComponents.splice(index + direction, 0, moved);
        setComponents(newComponents);
    };

    return (
        <div className="h-full flex gap-6">

            {/* 1. COMPONENT PALETTE */}
            <div className="w-64 flex-none space-y-4">
                <AppleCard title="Components" icon={LayoutDashboard} className="h-full">
                    <div className="space-y-2">
                        {COMPONENT_TYPES.map((item) => (
                            <div
                                key={item.type}
                                draggable
                                onDragStart={(e) => handleDragStart(e, item.type)}
                                className="flex items-center gap-3 p-3 bg-gray-50 hover:bg-white hover:shadow-md cursor-grab active:cursor-grabbing rounded-xl border border-transparent hover:border-gray-200 transition-all group"
                            >
                                <div className="p-2 bg-white rounded-lg border border-gray-100 group-hover:border-blue-100 group-hover:bg-blue-50 text-gray-500 group-hover:text-blue-500 transition-colors">
                                    <item.icon size={18} />
                                </div>
                                <span className="text-sm font-medium text-gray-700">{item.label}</span>
                            </div>
                        ))}
                    </div>
                </AppleCard>
            </div>

            {/* 2. CANVAS AREA */}
            <div className="flex-1 min-w-0">
                <div
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    className="h-full bg-gray-100/50 rounded-3xl border-2 border-dashed border-gray-300 hover:border-blue-400 hover:bg-blue-50/30 transition-all duration-300 overflow-y-auto p-8 relative"
                >
                    {components.length === 0 && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 pointer-events-none">
                            <LayoutDashboard size={64} className="mb-4 opacity-50" />
                            <p className="text-lg font-medium">Drag components here to build</p>
                        </div>
                    )}

                    <div className="space-y-4 max-w-4xl mx-auto">
                        {components.map((comp, index) => (
                            <div
                                key={comp.id}
                                onClick={() => setSelectedId(comp.id)}
                                className={`
                                    relative group rounded-2xl border-2 transition-all cursor-pointer
                                    ${selectedId === comp.id
                                        ? 'bg-white border-blue-500 shadow-xl scale-[1.02] z-10'
                                        : 'bg-white border-transparent hover:border-gray-200 shadow-apple'
                                    }
                                `}
                            >
                                {/* HOVER CONTROLS */}
                                <div className={`absolute -right-3 -top-3 flex gap-1 transition-opacity ${selectedId === comp.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
                                    {index > 0 && (
                                        <button onClick={(e) => { e.stopPropagation(); moveComponent(index, -1); }} className="p-1.5 bg-black text-white rounded-lg shadow-lg hover:bg-gray-800">
                                            <MoveUp size={14} />
                                        </button>
                                    )}
                                    {index < components.length - 1 && (
                                        <button onClick={(e) => { e.stopPropagation(); moveComponent(index, 1); }} className="p-1.5 bg-black text-white rounded-lg shadow-lg hover:bg-gray-800">
                                            <MoveDown size={14} />
                                        </button>
                                    )}
                                    <button onClick={(e) => { e.stopPropagation(); deleteComponent(comp.id); }} className="p-1.5 bg-red-500 text-white rounded-lg shadow-lg hover:bg-red-600">
                                        <Trash2 size={14} />
                                    </button>
                                </div>

                                {/* RENDERED COMPONENT */}
                                <div className="p-6 pointer-events-none">
                                    <h3 className="text-xs font-bold uppercase tracking-wider text-gray-400 mb-2">{comp.props.title || comp.type}</h3>

                                    {comp.type === 'gauge' && (
                                        <div className="flex items-end gap-2">
                                            <span className="text-4xl font-bold text-gray-900">{comp.props.value}</span>
                                            <span className="text-lg text-gray-500 mb-1">{comp.props.unit}</span>
                                        </div>
                                    )}

                                    {comp.type === 'stat' && (
                                        <div>
                                            <div className="text-3xl font-bold text-gray-900">{comp.props.value}</div>
                                            <div className="text-sm font-medium text-green-600 mt-1">{comp.props.trend}</div>
                                        </div>
                                    )}

                                    {comp.type === 'chart' && (
                                        <div className="w-full bg-gray-50 rounded-xl border border-gray-100 flex items-center justify-center text-gray-300" style={{ height: comp.props.height }}>
                                            <BarChart3 size={32} />
                                        </div>
                                    )}

                                    {comp.type === 'text' && (
                                        <p className="text-gray-600 leading-relaxed">{comp.props.content}</p>
                                    )}

                                    {comp.type === 'control' && (
                                        <div className="flex items-center justify-between bg-gray-50 p-3 rounded-xl">
                                            <span className="font-medium text-gray-700">{comp.props.label}</span>
                                            <div className="w-10 h-6 bg-green-500 rounded-full relative">
                                                <div className="absolute right-1 top-1 w-4 h-4 bg-white rounded-full shadow-sm" />
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* 3. PROPERTIES PANEL */}
            <div className="w-80 flex-none bg-white border-l border-gray-100 shadow-2xl relative z-20">
                <div className="h-full flex flex-col">
                    <div className="p-6 border-b border-gray-100">
                        <h2 className="text-lg font-bold flex items-center gap-2">
                            <Settings size={20} className="text-gray-400" />
                            Inspector
                        </h2>
                    </div>

                    <div className="flex-1 p-6 overflow-y-auto">
                        {selectedId ? (
                            <div className="space-y-6">
                                {Object.entries(components.find(c => c.id === selectedId).props).map(([key, val]) => (
                                    <div key={key}>
                                        <label className="block text-xs font-bold text-gray-400 uppercase tracking-wilder mb-2">{key}</label>
                                        <input
                                            type="text"
                                            value={val}
                                            onChange={(e) => updateProperty(key, e.target.value)}
                                            className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-xl focus:border-blue-500 focus:ring-2 focus:ring-blue-100 outline-none transition-all text-sm font-medium"
                                        />
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center text-gray-400 mt-10">
                                <p>Select a component to edit properties</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>

        </div>
    );
};

export default DashboardBuilder;
