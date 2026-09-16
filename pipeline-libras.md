# Pipeline de Reconhecimento de Sinais em Libras

## Objetivo

Construir um pipeline que recebe um vídeo de uma pessoa fazendo sinais em
Libras e devolve o significado do(s) sinal(is) executado(s), pra alimentar
uma camada seguinte do sistema (ex: montagem de frase em português, resposta
de um assistente, etc).

Fluxo de alto nível:

```
vídeo bruto
   │
   ▼
[1] Ingestão
   │
   ▼
[2] Pré-processamento
   │
   ▼
[3] Extração de pose / mãos / rosto (landmarks)
   │
   ▼
[4] Classificação do sinal (modelo)
   │
   ▼
[5] Anotação estruturada (saída)
   │
   ▼
camada de baixo (consome o significado)
```

---

## Dataset

Usamos o **MINDS-Libras** (repositório `ibmectech/minds-libras-raw` no
Hugging Face). É um dataset de reconhecimento de sinais isolados
(word-level): cada vídeo contém a execução de **um único sinal**.

- 800 vídeos `.mp4`, resolução 1920×1080
- 20 classes de sinais (Acontecer, Aluno, Amarelo, América, Aproveitar,
  Bala, Banco, Banheiro, Barulho, Cinco, Conhecer, Espelho, Esquina, Filho,
  Maca, Medo, Ruim, Sapo, Vacina, Vontade)
- 8 sinalizadores diferentes, 5 repetições por sinal/sinalizador
- `annotations.csv` mapeia cada vídeo (`video_id`) ao rótulo (`class`) e ao
  sinalizador (`user_id`)

**Importante:** esse dataset é só vídeo bruto + rótulo. Ele **não** contém
landmarks/pose/pontos faciais pré-extraídos — isso é responsabilidade do
nosso pipeline (etapa 3), não do dataset. Se em algum momento o repositório
disponibilizar uma versão com landmarks, **não usar**: o objetivo é o
pipeline aprender a extrair essas features sozinho, não depender de um
dataset que já vem processado.

---

## O que o script `download_minds_libras.py` já faz

Esse script já está pronto e resolve a etapa de aquisição do dataset:

1. Lista todos os arquivos do repositório `ibmectech/minds-libras-raw` no
   Hugging Face via `HfApi.list_repo_files`.
2. Filtra e baixa **apenas**:
   - `annotations.csv` (rótulos)
   - arquivos dentro de `videos/*.mp4` (vídeos brutos)
3. Tem uma lista de bloqueio por palavra-chave (`landmark`, `pose`, `joint`,
   `keypoint`, `skeleton`, `depth`, `holistic`, `mediapipe`) — qualquer
   arquivo com esses termos no nome é ignorado automaticamente, mesmo que o
   repositório adicione esse tipo de dado no futuro.
4. Salva tudo em `./dataset/minds-libras/` (estrutura `videos/` +
   `annotations.csv`), preservando a organização original.
5. Aceita `--repo` pra trocar de dataset (ex: `ibmectech/v-librasil-raw`) e
   `--output` pra mudar a pasta de destino.

Isso cobre só a etapa **[1] Ingestão** (aquisição) do pipeline. As etapas
[2] a [5] ainda precisam ser implementadas.

---

## Etapas que faltam implementar

### [2] Pré-processamento
- Ler o vídeo (`.mp4`) e extrair frames
- Padronizar fps e resolução entre vídeos de sinalizadores diferentes
- Opcional: recorte por intensidade de movimento, pra remover frames
  parados no início/fim do vídeo

### [3] Extração de pose / mãos / rosto
- Rodar um extrator de landmarks (ex: MediaPipe Holistic) frame a frame
- Gerar uma sequência de vetores de features por vídeo (coordenadas
  normalizadas de mãos, pose corporal e rosto)
- Persistir essa sequência em formato tabular (parquet/csv) associada ao
  `video_id`, sem perder o rótulo (`class`) vindo do `annotations.csv`

### [4] Classificação do sinal
- Treinar um modelo sobre as sequências de landmarks (ex: LSTM, GRU, CNN 3D
  ou Transformer) usando `class` como rótulo
- Validar com split por sinalizador (leave-one-signer-out), pra medir
  generalização de verdade e não só decorar o sinalizador

### [5] Anotação estruturada / saída
- Não repassar só o texto do sinal pra camada de baixo. Estruturar a saída
  com algo como:

```json
{
  "sinal": "Aluno",
  "confianca": 0.94,
  "inicio_ms": 320,
  "fim_ms": 1580,
  "sinalizador_id": "opcional, se disponível"
}
```

- Essa camada de saída é o contrato entre esse pipeline e a camada
  seguinte — deve ser estável mesmo se o modelo interno mudar.

---

## Pedido pro agente que for implementar

Modularizar isso bem certinho, com cada etapa isolada e testável
separadamente — nada de um script monolítico fazendo tudo de uma vez.
Sugestão de estrutura:

```
libras_pipeline/
├── ingestion/          # download + organização do dataset (já existe)
│   └── download_minds_libras.py
├── preprocessing/       # extração de frames, normalização
│   └── video_preprocessor.py
├── feature_extraction/  # MediaPipe Holistic / landmarks
│   └── landmark_extractor.py
├── model/                # treino e inferência do classificador
│   ├── train.py
│   └── predict.py
├── schema/               # contrato de saída (dataclass / pydantic)
│   └── sign_annotation.py
└── pipeline.py           # orquestra as etapas acima, ponta a ponta
```

Cada módulo deve:
- Ter uma responsabilidade única e clara
- Expor uma função/classe de entrada bem definida (input/output tipado)
- Não depender de detalhes internos dos outros módulos (só do contrato de
  entrada/saída entre eles)
- Ser testável isoladamente, sem precisar rodar o pipeline inteiro

O `pipeline.py` final deve conseguir receber um caminho de vídeo e devolver
o objeto de anotação estruturada (etapa [5]), chamando cada módulo em
sequência.
