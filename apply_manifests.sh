#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Starting deployment to Kubernetes..."

# 1. Создаем StorageClass
echo "Creating StorageClass..."
kubectl apply -f k8s/local-storage.yaml
sleep 2  # Даем время на создание StorageClass

# 2. Создаем namespace
echo "Creating namespace..."
kubectl apply -f k8s/namespace.yaml
sleep 2  # Даем время на создание namespace

# 3. Применяем ConfigMap
echo "Applying ConfigMap..."
kubectl apply -f k8s/configmap.yaml

# 4. Создаем PVC
echo "Creating Persistent Volume Claims..."
kubectl apply -f k8s/redis-pvc.yaml
kubectl apply -f k8s/app-pvc.yaml

# Ждем создания PVC
echo "Waiting for PVCs to be bound..."
echo -e "${YELLOW}Note: This might take some time as PVCs are in WaitForFirstConsumer mode${NC}"

# 5. Развертываем Redis
echo "Deploying Redis..."
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/redis-service.yaml

# Ждем готовности Redis
echo "Waiting for Redis deployment..."
kubectl -n mineral-classifier rollout status deployment/mineral-classifier-redis

# 6. Развертываем приложение
echo "Deploying application..."
kubectl apply -f k8s/app-deployment.yaml
kubectl apply -f k8s/app-service.yaml

# Ждем готовности приложения
echo "Waiting for application deployment..."
kubectl -n mineral-classifier rollout status deployment/mineral-classifier

echo -e "${GREEN}Deployment completed!${NC}"
echo "Service information:"
kubectl get service mineral-classifier -n mineral-classifier