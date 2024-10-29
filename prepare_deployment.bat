@echo off
setlocal enabledelayedexpansion

echo Starting deployment preparation...

:: Создаем временную директорию для деплоя
if not exist deployment_package mkdir deployment_package

:: Копируем Python файлы
echo Copying Python files...
copy *.py deployment_package\
copy requirements.txt deployment_package\
copy Dockerfile deployment_package\

:: Копируем директории
echo Copying directories...
if exist templates xcopy /E /I templates deployment_package\templates
if exist static xcopy /E /I static deployment_package\static
if exist data xcopy /E /I data deployment_package\data

:: Копируем Excel файл
echo Copying Excel file...
copy "Справочник_для_редактирования_09.10.2024.xlsx" deployment_package\

:: Копируем k8s манифесты
echo Copying Kubernetes manifests...
if not exist deployment_package\k8s mkdir deployment_package\k8s
copy k8s\*.yaml deployment_package\k8s\

:: Копируем скрипты развертывания
copy deploy.sh deployment_package\

:: Создаем архив (используя PowerShell для создания ZIP)
echo Creating archive...
powershell Compress-Archive -Path deployment_package\* -DestinationPath deployment_package.zip -Force

:: Очищаем временную директорию
echo Cleaning up...
rmdir /S /Q deployment_package

echo Package created: deployment_package.zip
echo Now copy this package to Linux server and run deploy.sh 