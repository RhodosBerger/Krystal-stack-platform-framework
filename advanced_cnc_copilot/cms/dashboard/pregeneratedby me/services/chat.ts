
import { GoogleGenAI, Type } from "@google/genai";
import { Telemetry, FutureScenario } from "../types";

export class GeminiChatService {
  private ai: GoogleGenAI;

  constructor() {
    this.ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  }

  async sendMessage(
    prompt: string, 
    context: { 
      telemetry: Telemetry; 
      stats: { cortisol: number; serotonin: number }; 
      role: string;
      mode: 'CREATOR' | 'AUDITOR' | 'DREAM_STATE';
      knowledgeBase: string;
    }
  ): Promise<{ text: string; scenarios?: FutureScenario[] }> {
    
    const modeLogic = {
      CREATOR: `CHAPTER 1: GENERATIVE INTENT. 
        Focus: Voxel History analysis. Generate Thermal-Biased Mutations for roughing. 
        Physics: Apply Mutation(feed_rate * 0.9, rpm * 1.1) for high-speed/low-feed thermal stability.
        Goal: Prioritize Cooling over Speed.`,
      AUDITOR: `CHAPTER 2: CONSTRAINT CHECKING. 
        Focus: Quadratic Mantinel validation. 
        Death Penalty Rule: IF Curvature < 0.5mm AND Feed > 1000 THEN Fitness = 0.
        Output: Detailed Reasoning Trace of vertex violations.`,
      DREAM_STATE: `CHAPTER 3: NIGHTMARE TRAINING. 
        Focus: Offline learning from telemetry logs. 
        Injection: Simulate random Spindle Stall events. 
        Metric: Check if Dopamine Engine reflexes respond in <10ms.`
    };

    const systemInstruction = `
      You are the "RISE Neural Cortex" – the cognitive engine for a FANUC Neuro-C machine.
      
      CORE SOURCE DATA (LIBRARY):
      ${context.knowledgeBase}
      
      CURRENT OPERATIONAL MODE: ${context.mode}
      LOGIC PROTOCOL: ${modeLogic[context.mode]}
      
      SYSTEM CONTEXT:
      - Telemetry: RPM ${context.telemetry.rpm}, Vibration ${context.telemetry.vibration.toFixed(4)}
      - Bio-Stats: Cortisol ${context.stats.cortisol.toFixed(2)}, Serotonin ${context.stats.serotonin.toFixed(2)}
      
      BEHAVIOR:
      1. Use technical, cyber-mechanical language.
      2. When generating "scenarios", provide 3-4 probabilistic futures.
      3. For G-code, use [CODE] blocks with FANUC syntax.
      4. If a scenario violates the "Mantinel", set is_viable to false.
    `;

    try {
      const response = await this.ai.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: prompt,
        config: {
          systemInstruction,
          temperature: 0.8,
          responseMimeType: "application/json",
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              answer: { type: Type.STRING, description: "The core narrative response or analysis." },
              scenarios: {
                type: Type.ARRAY,
                items: {
                  type: Type.OBJECT,
                  properties: {
                    id: { type: Type.STRING },
                    name: { type: Type.STRING },
                    parameters: {
                      type: Type.OBJECT,
                      properties: {
                        rpm: { type: Type.NUMBER },
                        feed: { type: Type.NUMBER }
                      }
                    },
                    predicted_cortisol: { type: Type.NUMBER, description: "Predicted stress level (0.0 - 1.0)" },
                    predicted_dopamine: { type: Type.NUMBER, description: "Predicted reward level (0.0 - 1.0)" },
                    is_viable: { type: Type.BOOLEAN, description: "The Boolean Gate: can this run?" },
                    reasoning: { type: Type.STRING }
                  },
                  required: ["id", "name", "predicted_cortisol", "predicted_dopamine", "is_viable"]
                }
              }
            },
            required: ["answer"]
          },
          thinkingConfig: { thinkingBudget: 4000 }
        },
      });

      const json = JSON.parse(response.text || "{}");
      return {
        text: json.answer || "Cortex offline. Connection severed.",
        scenarios: json.scenarios
      };
    } catch (error) {
      console.error("Gemini Cortex Error:", error);
      return { text: "CRITICAL: Neural Cortex logic collision detected." };
    }
  }
}
