"""
calculos_ar.py
================
Razões de aspecto (Aspect Ratios) geométricas sobre a malha facial do
MediaPipe, usadas para diagnosticar Expressões Não-Manuais (ENMs) em
Libras (sobrancelha, olho, boca, bochecha, nariz).

Portado de APL2-LIBRAS (testes/calculos_ar.py). As fórmulas e os
índices de landmark são os mesmos do projeto original — eles dependem
só da topologia da malha facial de 478 pontos do MediaPipe (FaceMesh),
que é idêntica entre a API legada (`mp.solutions.holistic`, usada no
projeto original) e a Tasks API (`HolisticLandmarker`) usada aqui, daí
funcionarem sem alteração sobre `result.face_landmarks` do
`HolisticLandmarker`. A diferença entre os dois projetos é só a fonte
dos landmarks (webcam ao vivo vs. frames de vídeo já gravado).

Cada landmark passado às funções abaixo só precisa expor `.x`, `.y` e
`.z` (coordenadas normalizadas em [0,1]), que é o contrato tanto do
`NormalizedLandmark` da API legada quanto do da Tasks API.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

# Fator de temperatura para a Softmax (valores maiores amortecem saltos bruscos)
TEMPERATURA = 1.5


# --- DISTÂNCIA EUCLIDIANA ROBUSTA (2D e 3D Real) ---

def ponto_3d_real(p: Any, w: float, h: float) -> np.ndarray:
    """Converte as coordenadas do MediaPipe (x, y em [0,1]) pra uma
    escala 3D metricamente coerente em pixels (z escalonado como x)."""
    return np.array([p.x * w, p.y * h, p.z * w])


def dist_3d(p1: Any, p2: Any, w: float, h: float) -> float:
    """Distância euclidiana 3D real no espaço de pixels/profundidade."""
    v1 = ponto_3d_real(p1, w, h)
    v2 = ponto_3d_real(p2, w, h)
    return float(np.linalg.norm(v1 - v2))


def dist_2d(p1: Any, p2: Any, w: float, h: float) -> float:
    """Distância euclidiana 2D simples."""
    x1, y1 = p1.x * w, p1.y * h
    x2, y2 = p2.x * w, p2.y * h
    return math.hypot(x2 - x1, y2 - y1)


# --- ESTIMATIVA ROBUSTA DE YAW (ROTAÇÃO DA CABEÇA) ---

def estimar_yaw(landmarks: Sequence[Any]) -> float:
    """Estima a rotação horizontal da cabeça comparando a posição do
    nariz com o contorno das bochechas (positivo = virado à direita)."""
    p_nariz = landmarks[1]
    p_esq = landmarks[234]
    p_dir = landmarks[454]

    dist_esq = abs(p_nariz.x - p_esq.x)
    dist_dir = abs(p_dir.x - p_nariz.x)
    total = dist_esq + dist_dir

    if total == 0:
        return 0.0
    return (dist_dir - dist_esq) / total


# --- BASE DE REFERÊNCIA INTEROCULAR 3D ---

def obter_escala_interocular_3d(landmarks: Sequence[Any], w: float, h: float) -> float:
    """Fator de escala de referência: média das distâncias 3D entre
    cantos externos e internos dos dois olhos."""
    d1 = dist_3d(landmarks[33], landmarks[263], w, h)  # Cantos externos
    d2 = dist_3d(landmarks[133], landmarks[362], w, h)  # Cantos internos
    return (d1 + d2) / 2.0


# --- 1. EAR (Eye Aspect Ratio - Abertura dos Olhos) ---

def calcular_ear(olho_landmarks: Sequence[int], landmarks: Sequence[Any], w: float, h: float) -> float:
    """EAR de Soukupová & Čech estendido com verificação multi-ponto.
    olho_landmarks: [v1_top, v1_bottom, v2_top, v2_bottom, h_left, h_right]"""
    v1 = dist_2d(landmarks[olho_landmarks[0]], landmarks[olho_landmarks[1]], w, h)
    v2 = dist_2d(landmarks[olho_landmarks[2]], landmarks[olho_landmarks[3]], w, h)
    h_dist = dist_2d(landmarks[olho_landmarks[4]], landmarks[olho_landmarks[5]], w, h)

    if h_dist < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)


# --- 2. MAR (Mouth Aspect Ratio - Abertura da Boca) ---

def calcular_mar(landmarks: Sequence[Any], w: float, h: float) -> float:
    """MAR com 3 vetores verticais internos + externos (estável no riso/fala)."""
    v1 = dist_2d(landmarks[13], landmarks[14], w, h)
    v2 = dist_2d(landmarks[82], landmarks[87], w, h)
    v3 = dist_2d(landmarks[312], landmarks[317], w, h)

    h_dist = dist_2d(landmarks[61], landmarks[291], w, h)

    if h_dist < 1e-6:
        return 0.0
    return (v1 + v2 + v3) / (3.0 * h_dist)


# --- 3. BAR (Brow Aspect Ratio - Altura e Franzimento 3D) ---

def calcular_bar(landmarks: Sequence[Any], w: float, h: float) -> tuple[float, float]:
    """Altura e aproximação das sobrancelhas, normalizadas pela
    distância interocular 3D (imune à pose da cabeça)."""
    escala = obter_escala_interocular_3d(landmarks, w, h)
    if escala < 1e-6:
        return 0.0, 0.0

    alt_esq = dist_3d(landmarks[70], landmarks[159], w, h)
    alt_dir = dist_3d(landmarks[300], landmarks[386], w, h)
    altura_relativa = ((alt_esq + alt_dir) / 2.0) / escala

    d_pontas = dist_3d(landmarks[55], landmarks[285], w, h)
    d_arcos = dist_3d(landmarks[107], landmarks[336], w, h)
    d_v_esq = dist_3d(landmarks[55], landmarks[9], w, h)
    d_v_dir = dist_3d(landmarks[285], landmarks[9], w, h)

    franzimento_composto = (d_pontas + d_arcos + d_v_esq + d_v_dir) / 4.0
    dist_relativa_juntas = franzimento_composto / escala

    return altura_relativa, dist_relativa_juntas


# --- 4. NMAR (Normalized Mouth Corner Ratio - Sorriso e Esticamento) ---

def calcular_nmar(landmarks: Sequence[Any], w: float, h: float) -> float:
    """Largura da boca normalizada pela distância 3D dos olhos."""
    escala = obter_escala_interocular_3d(landmarks, w, h)
    if escala < 1e-6:
        return 0.0

    dist_boca = dist_3d(landmarks[61], landmarks[291], w, h)
    return dist_boca / escala


# --- 5. PUP (Pucker Ratio - Bico / Lábios Projetados) ---

def calcular_pup(landmarks: Sequence[Any], w: float, h: float) -> float:
    """Projeção vertical dos lábios externos/internos vs largura labial."""
    largura_boca = dist_2d(landmarks[61], landmarks[291], w, h)
    if largura_boca < 1e-6:
        return 0.0

    alt_ext = dist_2d(landmarks[0], landmarks[17], w, h)
    alt_int = dist_2d(landmarks[13], landmarks[14], w, h)

    return ((alt_ext + alt_int) / 2.0) / largura_boca


# --- 6. CAR (Cheek Aspect Ratio - Bochechas Infladas) ---

def calcular_car(landmarks: Sequence[Any], w: float, h: float) -> float:
    """Expansão das bochechas normalizada pelo eixo vertical central do rosto."""
    alt_rosto1 = dist_3d(landmarks[1], landmarks[152], w, h)
    alt_rosto2 = dist_3d(landmarks[10], landmarks[152], w, h)
    alt_media = (alt_rosto1 + alt_rosto2 / 2.0) / 2.0

    if alt_media < 1e-6:
        return 0.0

    larg_bochechas1 = dist_3d(landmarks[234], landmarks[454], w, h)
    larg_bochechas2 = dist_3d(landmarks[192], landmarks[416], w, h)
    larg_media = (larg_bochechas1 + larg_bochechas2) / 2.0

    return larg_media / (2.0 * alt_media)


# --- 7. NAR (Nasal Aspect Ratio - Nariz Franzido) ---

def calcular_nar(landmarks: Sequence[Any], w: float, h: float) -> float:
    """Encurtamento do dorso nasal (AU9 - Nose Wrinkle)."""
    dist_interna_olhos = dist_3d(landmarks[133], landmarks[362], w, h)
    if dist_interna_olhos < 1e-6:
        return 0.0

    dorso1 = dist_3d(landmarks[198], landmarks[2], w, h)
    dorso2 = dist_3d(landmarks[6], landmarks[2], w, h)
    dorso_medio = (dorso1 + dorso2) / 2.0

    return dorso_medio / dist_interna_olhos


# --- SOFTMAX COM TEMPERATURA E PROTEÇÃO NUMÉRICA ---

def calcular_probabilidades(scores: Sequence[float]) -> np.ndarray:
    """Softmax com escala de temperatura ajustada (TEMPERATURA)."""
    scores_array = np.array(scores, dtype=np.float64)
    scaled = scores_array / TEMPERATURA
    exp_scores = np.exp(scaled - np.max(scaled))

    soma = exp_scores.sum()
    if soma == 0:
        return np.ones_like(scores_array) / len(scores_array)

    return exp_scores / soma


# --- Pontos usados pro EAR esquerdo/direito (ver rotulador.py original) ---
PONTOS_OLHO_ESQ: tuple[int, ...] = (160, 144, 158, 153, 33, 133)
PONTOS_OLHO_DIR: tuple[int, ...] = (385, 380, 387, 373, 362, 263)


def calcular_todos_ratios(landmarks: Sequence[Any], w: float, h: float) -> dict[str, float]:
    """Calcula de uma vez todos os Aspect Ratios faciais usados pro
    diagnóstico de ENMs, a partir da malha facial de um único frame."""
    ear_esq = calcular_ear(PONTOS_OLHO_ESQ, landmarks, w, h)
    ear_dir = calcular_ear(PONTOS_OLHO_DIR, landmarks, w, h)
    bar_altura, bar_juntas = calcular_bar(landmarks, w, h)

    return {
        "ear_esq": ear_esq,
        "ear_dir": ear_dir,
        "ear_medio": (ear_esq + ear_dir) / 2.0,
        "mar": calcular_mar(landmarks, w, h),
        "bar_altura": bar_altura,
        "bar_juntas": bar_juntas,
        "nmar": calcular_nmar(landmarks, w, h),
        "pup": calcular_pup(landmarks, w, h),
        "car": calcular_car(landmarks, w, h),
        "nar": calcular_nar(landmarks, w, h),
        "yaw": estimar_yaw(landmarks),
    }
