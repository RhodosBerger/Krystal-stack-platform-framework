/** @type {import('tailwindcss').Config} */
import tokens from './src/design-tokens.json';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        industrial: tokens.theme.colors.industrial,
        neuro: tokens.theme.colors.neuro,
        hologram: tokens.theme.colors.hologram,
      },
      fontFamily: {
        sans: tokens.theme.typography.sans.split(', '),
        mono: tokens.theme.typography.mono.split(', '),
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        }
      }
    },
  },
  plugins: [],
}
