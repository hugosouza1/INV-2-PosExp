#!/bin/bash

set -e

echo "========================================"
echo " Instalando Crow C++"
echo "========================================"

# ------------------------------------------------------------
# 1. Atualizar pacotes
# ------------------------------------------------------------

echo "[1/6] Atualizando pacotes..."
sudo apt update


# ------------------------------------------------------------
# 2. Dependências
# ------------------------------------------------------------

echo "[2/6] Instalando dependências..."

sudo apt install -y \
    git \
    build-essential \
    cmake \
    libasio-dev \
    libboost-all-dev


# ------------------------------------------------------------
# 3. Diretório temporário
# ------------------------------------------------------------

echo "[3/6] Preparando diretório..."

TEMP_DIR="/tmp/crow-install"

rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

cd "$TEMP_DIR"


# ------------------------------------------------------------
# 4. Baixar Crow
# ------------------------------------------------------------

echo "[4/6] Baixando Crow..."

git clone --depth 1 https://github.com/CrowCpp/Crow.git

cd Crow


# ------------------------------------------------------------
# 5. Compilar
# ------------------------------------------------------------

echo "[5/6] Compilando Crow..."

cmake -S . -B build \
    -DCROW_BUILD_EXAMPLES=OFF \
    -DCROW_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j"$(nproc)"


# ------------------------------------------------------------
# 6. Instalar
# ------------------------------------------------------------

echo "[6/6] Instalando Crow..."

sudo cmake --install build


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

echo ""
echo "========================================"
echo " Crow instalado com sucesso!"
echo "========================================"

echo ""
echo "Verificando instalação..."

if [ -d "/usr/local/include/crow" ]; then
    echo "Headers encontrados em:"
    echo "/usr/local/include/crow"
else
    echo "AVISO: diretório de headers não encontrado."
fi

echo ""
echo "Pronto."