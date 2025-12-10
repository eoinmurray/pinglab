import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    path.join(__dirname, "../biblio/**/*.mdx"),
    path.join(__dirname, "../biblio/**/*.md"),
  ],
  theme: {
  	extend: {
  		fontFamily: {
  			// Clean sans-serif for everything
  			sans: ['Inter', 'system-ui', 'sans-serif'],
  			// Monospace for code, data, technical elements
  			mono: ['JetBrains Mono', 'ui-monospace', 'monospace'],
  		},
  		fontSize: {
  			// Refined type scale
  			'xs': ['0.75rem', { lineHeight: '1.5' }],
  			'sm': ['0.875rem', { lineHeight: '1.6' }],
  			'base': ['1rem', { lineHeight: '1.75' }],
  			'lg': ['1.125rem', { lineHeight: '1.7' }],
  			'xl': ['1.25rem', { lineHeight: '1.5' }],
  			'2xl': ['1.5rem', { lineHeight: '1.35' }],
  			'3xl': ['1.875rem', { lineHeight: '1.25' }],
  			'4xl': ['2.25rem', { lineHeight: '1.2' }],
  			'5xl': ['3rem', { lineHeight: '1.15' }],
  			'6xl': ['3.75rem', { lineHeight: '1.1' }],
  		},
  		letterSpacing: {
  			'tightest': '-0.04em',
  			'tighter': '-0.02em',
  			'tight': '-0.01em',
  			'normal': '0',
  			'wide': '0.01em',
  			'wider': '0.02em',
  			'widest': '0.05em',
  		},
  		borderRadius: {
  			lg: 'var(--radius)',
  			md: 'calc(var(--radius) - 2px)',
  			sm: 'calc(var(--radius) - 4px)'
  		},
  		colors: {
  			// Ministry of Unsanctioned Computation
  			ministry: {
  				parchment: '#F5F0E6',
  				cream: '#FAF7F0',
  				crimson: '#8B1538',
  				gold: '#B8860B',
  				brown: '#5C4033',
  				ink: '#2C1810',
  			},
  			background: 'hsl(var(--background))',
  			foreground: 'hsl(var(--foreground))',
  			card: {
  				DEFAULT: 'hsl(var(--card))',
  				foreground: 'hsl(var(--card-foreground))'
  			},
  			popover: {
  				DEFAULT: 'hsl(var(--popover))',
  				foreground: 'hsl(var(--popover-foreground))'
  			},
  			primary: {
  				DEFAULT: 'hsl(var(--primary))',
  				foreground: 'hsl(var(--primary-foreground))'
  			},
  			secondary: {
  				DEFAULT: 'hsl(var(--secondary))',
  				foreground: 'hsl(var(--secondary-foreground))'
  			},
  			muted: {
  				DEFAULT: 'hsl(var(--muted))',
  				foreground: 'hsl(var(--muted-foreground))'
  			},
  			accent: {
  				DEFAULT: 'hsl(var(--accent))',
  				foreground: 'hsl(var(--accent-foreground))'
  			},
  			destructive: {
  				DEFAULT: 'hsl(var(--destructive))',
  				foreground: 'hsl(var(--destructive-foreground))'
  			},
  			border: 'hsl(var(--border))',
  			input: 'hsl(var(--input))',
  			ring: 'hsl(var(--ring))',
  			chart: {
  				'1': 'hsl(var(--chart-1))',
  				'2': 'hsl(var(--chart-2))',
  				'3': 'hsl(var(--chart-3))',
  				'4': 'hsl(var(--chart-4))',
  				'5': 'hsl(var(--chart-5))'
  			},
  			sidebar: {
  				DEFAULT: 'hsl(var(--sidebar-background))',
  				foreground: 'hsl(var(--sidebar-foreground))',
  				primary: 'hsl(var(--sidebar-primary))',
  				'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
  				accent: 'hsl(var(--sidebar-accent))',
  				'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
  				border: 'hsl(var(--sidebar-border))',
  				ring: 'hsl(var(--sidebar-ring))'
  			}
  		},
  		animation: {
  			'fade-in': 'fade-in 0.4s ease-out forwards',
  			'fade-in-slow': 'fade-in-slow 0.6s ease-out forwards',
  			'slide-up': 'slide-up 0.5s ease-out forwards',
  		},
  		keyframes: {
  			'fade-in': {
  				from: { opacity: '0', transform: 'translateY(8px)' },
  				to: { opacity: '1', transform: 'translateY(0)' },
  			},
  			'fade-in-slow': {
  				from: { opacity: '0' },
  				to: { opacity: '1' },
  			},
  			'slide-up': {
  				from: { opacity: '0', transform: 'translateY(20px)' },
  				to: { opacity: '1', transform: 'translateY(0)' },
  			},
  		},
  		transitionTimingFunction: {
  			'out-expo': 'cubic-bezier(0.19, 1, 0.22, 1)',
  		},
  	}
  },
  plugins: [require("tailwindcss-animate"), require("@tailwindcss/typography")],
}
