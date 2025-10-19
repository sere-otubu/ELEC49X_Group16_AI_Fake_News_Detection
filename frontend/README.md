# Frontend - Fake News Detection UI

React + Vite + Chakra UI frontend for the fake news detection app.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at http://localhost:5173

## Features

- 🎨 Modern, clean UI with Chakra UI
- 📱 Responsive design
- 🚀 Fast development with Vite
- ⚡ Real-time analysis feedback
- 📊 Visual percentage bar for results
- 🎯 Color-coded results
- 🔔 Toast notifications

## Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Configuration

The backend API URL is configured in `src/App.jsx`:

```javascript
const API_URL = 'http://localhost:8000'
```

Change this if your backend runs on a different port.

## Building for Production

```bash
npm run build
```

The optimized files will be in the `dist/` directory.

## Dependencies

- **React** - UI library
- **Chakra UI** - Component library
- **Axios** - HTTP client
- **Vite** - Build tool
- **Framer Motion** - Animations (Chakra UI dependency)
