
export interface Telemetry {
  speed: number;
  quality: number;
  vibration: number;
  temp: number;
  rpm: number;
  spindleLoad: number;
  dopamine_score?: number;
  cortisol_level?: number;
}

export interface Job {
  id: string;
  name: string;
  mass: number; 
  velocity: number; 
}

export interface Machine {
  id: string;
  name: string;
  gravity: number; 
  status: 'idle' | 'running' | 'warning' | 'error';
  activeJobId?: string;
  toolWear: number;
}

export interface Material {
  name: string;
  maxPower: number; // Watts
  thermalLimit: number; // °C
  ductility: number;
}

export interface User {
  id: string;
  username: string;
  role: UserRole;
  token?: string;
}

export enum SimulationMode {
  IDLE = 'IDLE',
  RUSH = 'RUSH',
  ECONOMY = 'ECONOMY'
}

export interface FutureScenario {
  id: string;
  name: string;
  parameters: { rpm: number; feed: number };
  predicted_cortisol: number;
  predicted_dopamine: number;
  is_viable: boolean;
  reasoning: string;
}

export interface Message {
  id: string;
  role: 'user' | 'creator' | 'auditor' | 'dream_state';
  text: string;
  status?: 'approved' | 'veto' | 'pending';
  reasoning?: string;
  scenarios?: FutureScenario[];
}

export interface CouncilState {
  messages: Message[];
  isLocked: boolean;
  canRun: boolean;
}

export type CouncilAction = 
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'SET_LOCK'; locked: boolean }
  | { type: 'RESET' };

export interface AuditLog {
  timestamp: string;
  userHash: string;
  diff: string;
  version: string;
}

export interface TranscriptionPart {
  text: string;
  role: 'user' | 'model';
  timestamp: number;
}

export interface MarketplaceScript {
  id: string;
  name: string;
  survivorScore: number;
  runs: number;
  avgVibration: number;
}

export type UserRole = 'OPERATOR' | 'MANAGER' | 'CREATOR' | 'ADMIN';
