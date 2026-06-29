# 多阶段构建：pnpm 打包静态资源 → nginx 托管
FROM node:22-alpine AS builder
WORKDIR /app
RUN npm install -g pnpm
COPY package.json pnpm-lock.yaml* pnpm-workspace.yaml ./
RUN pnpm install
COPY . .
RUN pnpm build

FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
