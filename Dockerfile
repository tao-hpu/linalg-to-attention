# 多阶段构建：pnpm 打包静态资源 → nginx 托管
FROM node:22-alpine AS builder
WORKDIR /app
# 钉 pnpm 9：默认会执行依赖的构建脚本（esbuild 需要），不像 pnpm 10 那样默认拦截
RUN npm install -g pnpm@9
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
# 锁定 lockfile：同一个提交在任何时候都装到同一组依赖。
# 若这里因 lockfile 不同步而失败，正确做法是在本地跑 pnpm install 并提交更新后的
# pnpm-lock.yaml，而不是在镜像里绕开校验。
RUN pnpm install --frozen-lockfile
COPY . .
RUN pnpm build

FROM nginx:1.27-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
