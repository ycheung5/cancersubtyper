# ✅ Use Node.js 20+ for latest TailwindCSS/DaisyUI support
FROM node:20-alpine

# Set working directory inside the container
WORKDIR /app

# Copy package.json and package-lock.json first for caching
COPY package.json package-lock.json ./

# Install dependencies
RUN npm install

# Copy all project files (except ones in .dockerignore)
COPY . .

# Expose port 5173 for Vite
EXPOSE 5173

# Use environment variable from docker-compose
ENV CHOKIDAR_USEPOLLING=${CHOKIDAR_USEPOLLING:-true}

# Start the Vite development server with hot reloading
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
