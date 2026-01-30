# Dockerfile for Vue Frontend
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY frontend-vue/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY frontend-vue/ .

# Build the application
ENV NODE_ENV=production
RUN npm run build

# Production stage
FROM nginx:alpine AS production

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx-vue.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]