import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Primary palette - matching Streamlit dark theme
        background: {
          DEFAULT: '#0e1117',
          secondary: '#1e293b',
        },
        primary: {
          DEFAULT: '#3b82f6',
          dark: '#2563eb',
          darker: '#1d4ed8',
        },
        accent: {
          DEFAULT: '#00d4aa',
          dark: '#00b894',
        },
        surface: '#334155',
        // Semantic colors
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        // Text colors
        'text-primary': '#e2e8f0',
        'text-secondary': '#94a3b8',
        'text-muted': '#64748b',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'monospace'],
      },
      boxShadow: {
        card: '0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -2px rgba(0, 0, 0, 0.2)',
        'card-hover': '0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -4px rgba(0, 0, 0, 0.3)',
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
        'gradient-card': 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)',
        'gradient-accent': 'linear-gradient(135deg, #00d4aa 0%, #00b894 100%)',
      },
    },
  },
  plugins: [],
};

export default config;
