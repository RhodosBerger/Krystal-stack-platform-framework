
/**
 * FANUC RISE Design Tokens v2.1
 * Implements biological metaphors for industrial UI
 */

export const THEME = {
  colors: {
    safety: '#ff5722',
    emerald: '#10b981',
    cyber: '#3b82f6',
    zinc: {
      900: '#09090b',
      800: '#18181b'
    }
  },
  physics: {
    // Heartbeat pulse frequency tied to stress level
    getHeartbeat: (stress: number) => ({
      scale: [1, 1.02 + stress * 0.05, 1],
      opacity: [0.3, 0.8 + stress * 0.2, 0.3],
      transition: {
        duration: Math.max(0.2, 1.5 - stress * 1.3),
        repeat: Infinity,
        ease: "easeInOut" as const
      }
    }),
    // Synesthesia Entropy: Blur and Jitter based on vibration
    getEntropy: (vibration: number) => ({
      filter: `blur(${vibration * 10}px)`,
      x: vibration > 0.05 ? [0, -1, 1, 0] : 0,
      transition: {
        duration: 0.1,
        repeat: Infinity
      }
    })
  }
};

export type UserRole = 'OPERATOR' | 'MANAGER' | 'CREATOR';
