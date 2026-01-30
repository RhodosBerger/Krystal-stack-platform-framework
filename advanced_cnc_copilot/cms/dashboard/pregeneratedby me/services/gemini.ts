
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';

export interface VoiceCallbacks {
  onTranscription: (text: string, role: 'user' | 'model') => void;
  onStatusChange: (status: 'connecting' | 'open' | 'closed' | 'error') => void;
}

export class GeminiVoiceService {
  private sessionPromise: Promise<any> | null = null;
  private audioContext: AudioContext | null = null;
  private nextStartTime: number = 0;
  private sources: Set<AudioBufferSourceNode> = new Set();
  private inputTranscription: string = '';
  private outputTranscription: string = '';

  constructor(private callbacks: VoiceCallbacks) {
    // Guidelines: Always create GoogleGenAI instance right before making an API call.
  }

  async connect() {
    this.callbacks.onStatusChange('connecting');
    // Guidelines: Use process.env.API_KEY directly and instantiate here.
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    
    // AudioContext initialization follows recommended practice for cross-browser support
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    const inputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
    
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      console.error("Microphone access denied", e);
      this.callbacks.onStatusChange('error');
      return;
    }

    this.sessionPromise = ai.live.connect({
      model: 'gemini-2.5-flash-native-audio-preview-12-2025',
      callbacks: {
        onopen: () => {
          this.callbacks.onStatusChange('open');
          const source = inputAudioContext.createMediaStreamSource(stream);
          const scriptProcessor = inputAudioContext.createScriptProcessor(4096, 1, 1);
          scriptProcessor.onaudioprocess = (e) => {
            const inputData = e.inputBuffer.getChannelData(0);
            const pcmBlob = this.createBlob(inputData);
            // Guidelines: Rely solely on sessionPromise resolves to send input and prevent race conditions.
            this.sessionPromise?.then(session => session.sendRealtimeInput({ media: pcmBlob }));
          };
          source.connect(scriptProcessor);
          scriptProcessor.connect(inputAudioContext.destination);
        },
        onmessage: async (message: LiveServerMessage) => {
          if (message.serverContent?.outputTranscription) {
            this.outputTranscription += message.serverContent.outputTranscription.text;
            this.callbacks.onTranscription(message.serverContent.outputTranscription.text, 'model');
          } else if (message.serverContent?.inputTranscription) {
            this.inputTranscription += message.serverContent.inputTranscription.text;
            this.callbacks.onTranscription(message.serverContent.inputTranscription.text, 'user');
          }

          if (message.serverContent?.turnComplete) {
            this.inputTranscription = '';
            this.outputTranscription = '';
          }

          // Guidelines: Process model audio bytes using the parts[0] reference.
          const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
          if (base64Audio && this.audioContext) {
            this.nextStartTime = Math.max(this.nextStartTime, this.audioContext.currentTime);
            const buffer = await this.decodeAudioData(this.decode(base64Audio), this.audioContext, 24000, 1);
            const source = this.audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(this.audioContext.destination);
            source.addEventListener('ended', () => this.sources.delete(source));
            
            // Scheduling each new audio chunk to start at nextStartTime ensures smooth, gapless playback.
            source.start(this.nextStartTime);
            this.nextStartTime += buffer.duration;
            this.sources.add(source);
          }

          if (message.serverContent?.interrupted) {
            this.sources.forEach(s => s.stop());
            this.sources.clear();
            this.nextStartTime = 0;
          }
        },
        onerror: (e: ErrorEvent) => {
          console.error("Gemini Error", e);
          this.callbacks.onStatusChange('error');
        },
        onclose: (e: CloseEvent) => {
          this.callbacks.onStatusChange('closed');
        }
      },
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
        systemInstruction: "You are the Neuro-C Cognitive Core. You help operators manage advanced manufacturing telemetry, stress levels (cortisol), and gravitational scheduling. Be technical yet helpful. Use manufacturing physics metaphors.",
        outputAudioTranscription: {},
        inputAudioTranscription: {}
      }
    });
  }

  // Guidelines: Implement manual PCM blob creation with correct MIME type.
  private createBlob(data: Float32Array): Blob {
    const int16 = new Int16Array(data.length);
    for (let i = 0; i < data.length; i++) int16[i] = data[i] * 32768;
    return { data: this.encode(new Uint8Array(int16.buffer)), mimeType: 'audio/pcm;rate=16000' };
  }

  // Guidelines: Implement manual base64 encoding/decoding.
  private encode(bytes: Uint8Array) {
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  private decode(base64: string) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) bytes[i] = binaryString.charCodeAt(i);
    return bytes;
  }

  // Guidelines: Manual raw PCM decoding as AudioContext.decodeAudioData is for file formats only.
  private async decodeAudioData(data: Uint8Array, ctx: AudioContext, sampleRate: number, numChannels: number): Promise<AudioBuffer> {
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);
    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = buffer.getChannelData(channel);
      for (let i = 0; i < frameCount; i++) channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
    return buffer;
  }
}
