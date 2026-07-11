FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Build Next.js application
RUN npm run build

# Start the application
CMD ["npm", "start"]
