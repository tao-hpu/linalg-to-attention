#!/bin/bash
# 在 aws-hk 服务器上运行：拉取最新代码 → 重建镜像 → 重启容器
set -e

echo "==> Deploying linalg-to-attention..."

git fetch --all
git reset --hard origin/master
git pull

export DOCKER_BUILDKIT=1
docker compose down --remove-orphans 2>/dev/null || true
docker build -t linalg-to-attention:latest .
docker compose up -d

echo "==> Done. Serving on http://127.0.0.1:5191 (reverse-proxy this to your domain)"
