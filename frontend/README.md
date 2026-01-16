# Sri Lanka Trip Planner - Frontend

A React + TypeScript single-page application for the Sri Lanka Trip Planner.

## Features

- 🎨 Modern, responsive UI
- ⚡ Built with Vite for fast development
- 🔷 TypeScript for type safety
- 🐳 Docker support for easy deployment

## Getting Started

### Prerequisites

- Node.js 18 or higher
- npm or yarn

### Development

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Docker

### Build the Docker image:

```bash
docker build -t trip-planner-frontend .
```

### Run the container:

```bash
docker run -p 8080:80 trip-planner-frontend
```

The app will be available at `http://localhost:8080`

### Using Docker Compose (with backend):

Create a `docker-compose.yml` in the root directory and run:

```bash
docker-compose up
```

## Configuration

Update the API endpoint in `src/App.tsx`:

