#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Starting deployment process..."

# Проверяем наличие необходимых утилит
for cmd in kubectl docker unzip; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        exit 1
    fi
done

# Проверяем подключение к кластеру Kubernetes
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

# Проверяем наличие архива
if [ ! -f "deployment_package.zip" ]; then
    echo -e "${RED}Error: deployment_package.zip not found${NC}"
    exit 1
fi

# Создаем рабочую директорию
WORK_DIR="mineral_classifier_deployment"
rm -rf $WORK_DIR
mkdir -p $WORK_DIR

# Распаковываем архив
echo "Extracting deployment package..."
unzip -q deployment_package.zip -d $WORK_DIR

# Переходим в рабочую директорию
cd $WORK_DIR

# Проверяем Docker login
echo "Checking Docker Hub credentials..."
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

# Проверяем авторизацию в Docker Hub (изменённая часть)
echo "Checking Docker Hub authentication..."
if ! docker info | grep "Username: gimmyhat" &> /dev/null; then
    echo -e "${YELLOW}Warning: Not logged in to Docker Hub${NC}"
    echo "Attempting to login..."
    if ! docker login --username gimmyhat; then
        echo -e "${RED}Error: Docker Hub login failed${NC}"
        exit 1
    fi
fi

echo "Docker Hub authentication successful"

# Сборка и публикация образа
echo "Building and pushing Docker image..."
VERSION=$(date +%Y%m%d_%H%M%S)
if ! docker build -t gimmyhat/mineral-classifier:$VERSION .; then
    echo -e "${RED}Error: Docker build failed${NC}"
    exit 1
fi

docker tag gimmyhat/mineral-classifier:$VERSION gimmyhat/mineral-classifier:latest
docker push gimmyhat/mineral-classifier:$VERSION
docker push gimmyhat/mineral-classifier:latest

# Применяем манифесты Kubernetes
echo "Applying Kubernetes manifests..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml

# Создаем PVC
echo "Creating persistent volumes..."
kubectl apply -f k8s/redis-pvc.yaml
kubectl apply -f k8s/app-pvc.yaml

# Ждем создания PVC
echo "Waiting for PVC creation..."
kubectl wait --for=condition=Bound pvc/redis-pvc -n mineral-classifier --timeout=60s
kubectl wait --for=condition=Bound pvc/app-data-pvc -n mineral-classifier --timeout=60s
kubectl wait --for=condition=Bound pvc/app-results-pvc -n mineral-classifier --timeout=60s

# Развертываем Redis
echo "Deploying Redis..."
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/redis-service.yaml

# Ждем готовности Redis
echo "Waiting for Redis deployment..."
kubectl rollout status deployment/mineral-classifier-redis -n mineral-classifier

# Развертываем приложение
echo "Deploying application..."
kubectl apply -f k8s/app-deployment.yaml
kubectl apply -f k8s/app-service.yaml

# Ждем готовности приложения
echo "Waiting for application deployment..."
kubectl rollout status deployment/mineral-classifier -n mineral-classifier

# Получаем информацию о сервисе
echo -e "\n${GREEN}Deployment completed successfully!${NC}"
echo "Service information:"
kubectl get service mineral-classifier -n mineral-classifier

# Очистка
cd ..
rm -rf $WORK_DIR

echo -e "\n${GREEN}Deployment process completed!${NC}" 