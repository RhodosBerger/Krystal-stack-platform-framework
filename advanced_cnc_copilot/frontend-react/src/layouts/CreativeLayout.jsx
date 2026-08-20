import React from 'react';
import CreativeTwinPanel from '../components/CreativeTwinPanel';
import EmotionalNexus from '../components/EmotionalNexus';
import MaterialSelector from '../components/MaterialSelector';
import AssemblyCanvas from '../components/AssemblyCanvas';
import VRTrainingPanel from '../components/VRTrainingPanel';
import RoboticsSim from '../components/RoboticsSim';
import AppleCard from '../components/AppleCard';
import { usePersona } from '../context/PersonaContext';
import { Sparkles, Box, LayoutGrid, Layers, Wand2 } from 'lucide-react';

const CreativeLayout = () => {
   const { config } = usePersona();

   return (
      <div className="h-screen w-full bg-[#f2f2f7] text-gray-900 flex flex-col font-sans overflow-hidden relative">

         {/* 1. Floating Header (Dynamic Island Style) */}
         <div className="absolute top-6 left-1/2 -translate-x-1/2 z-30">
            <div className="bg-black text-white px-6 py-3 rounded-full shadow-2xl flex items-center gap-6">
               <div className="flex items-center gap-2">
                  <Sparkles size={16} className="text-yellow-400" />
                  <span className="font-semibold text-sm tracking-wide">Studio Gen-1</span>
               </div>
               <div className="w-px h-4 bg-white/20" />
               <div className="flex gap-4 text-xs font-medium text-gray-300">
                  <span className="hover:text-white cursor-pointer">Topology</span>
                  <span className="hover:text-white cursor-pointer text-white">Materials</span>
                  <span className="hover:text-white cursor-pointer">Rendering</span>
               </div>
            </div>
         </div>

         {/* 2. Main Workspace */}
         <div className="flex-1 grid grid-cols-12 gap-6 p-6 pt-24 h-full">

            {/* LEFT SIDEBAR: TOOLS */}
            <div className="col-span-12 lg:col-span-3 flex flex-col gap-6 h-full pointer-events-none">
               <div className="pointer-events-auto flex-none">
                  <AppleCard title="Substrate" icon={Box} className="shadow-lg">
                     <MaterialSelector />
                  </AppleCard>
               </div>

               <div className="pointer-events-auto flex-1 min-h-0">
                  <AppleCard title="Emotional Nexus" icon={Wand2} className="h-full shadow-lg">
                     <div className="h-full overflow-y-auto -mx-2 px-2">
                        <EmotionalNexus />
                     </div>
                  </AppleCard>
               </div>
            </div>

            {/* CENTER CANVAS (The Stage) */}
            <div className="col-span-12 lg:col-span-6 h-full flex flex-col gap-6 relative">
               <div className="flex-1 bg-white rounded-[2.5rem] shadow-[0_20px_40px_rgba(0,0,0,0.06)] border border-white relative overflow-hidden group">
                  <div className="absolute inset-0 bg-grid-slate-100/[0.04] bg-[length:32px_32px]" />

                  {/* The Assembly Canvas */}
                  <AssemblyCanvas assembled={[]} setAssembled={() => { }} />

                  {/* Floating Overlay Controls */}
                  <div className="absolute bottom-8 left-1/2 -translate-x-1/2 flex gap-2">
                     <button className="px-4 py-2 bg-white/90 backdrop-blur shadow-lg rounded-full text-xs font-bold hover:scale-105 transition-all text-gray-700">RESET VIEW</button>
                     <button className="px-4 py-2 bg-black text-white shadow-lg rounded-full text-xs font-bold hover:scale-105 transition-all">GENERATE MESH</button>
                  </div>
               </div>

               <div className="h-64">
                  <AppleCard className="h-full" title="Neural Topology" icon={LayoutGrid}>
                     <RoboticsSim />
                  </AppleCard>
               </div>
            </div>

            {/* RIGHT SIDEBAR: SIMULATION */}
            <div className="col-span-12 lg:col-span-3 flex flex-col gap-6 h-full pointer-events-none">
               <div className="pointer-events-auto h-1/2">
                  <AppleCard title="VR Training" icon={Layers} className="h-full shadow-lg">
                     <VRTrainingPanel />
                  </AppleCard>
               </div>

               <div className="pointer-events-auto h-1/2">
                  <AppleCard title="Digital Twin" icon={Sparkles} className="h-full shadow-lg">
                     <CreativeTwinPanel />
                  </AppleCard>
               </div>
            </div>

         </div>
      </div>
   );
};

export default CreativeLayout;
