#!/bin/bash

# Версия образа
VERSION=1.0.0

# Сборка образа
docker build -t gimmyhat/mineral-classifier:$VERSION .
docker tag gimmyhat/mineral-classifier:$VERSION gimmyhat/mineral-classifier:latest

# Публикация образа
docker push gimmyhat/mineral-classifier:$VERSION
docker push gimmyhat/mineral-classifier:latest 