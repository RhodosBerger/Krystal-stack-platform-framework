import axios from 'axios';

/**
 * Synaptic Bridge Client 🧠
 * Connects the Frontend "Creative Intent" to the Backend "Synaptic API".
 */
export const SynapticClient = {
    /**
     * Send a creative intent to the Synaptic Bridge.
     * @param {string} intent - Natural language description (e.g., "Make it angry")
     * @param {string} prompt - Technical prompt (e.g., "Milling Job 101")
     * @returns {Promise<Object>} - The generated protocol and detected emotion.
     */
    async createProtocol(intent, prompt) {
        try {
            const response = await axios.post('/api/synaptic/create', {
                intent: intent,
                prompt: prompt,
                name: "Synaptic_Job_" + Date.now().toString().slice(-4)
            });
            return response.data;
        } catch (error) {
            console.error("[SynapticClient] Creation Failed:", error);
            throw error;
        }
    }
};
