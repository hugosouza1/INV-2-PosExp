# Relatório de acurácia — 100 épocas, stride 2

Validação leave-one-signer-out (8 sinalizadores, 800 amostras de teste, 20 classes).
Configuração: epochs=100, batch_size=8, lr=0.001, augment=False, n_samples=800, n_classes=20, frame_stride=2

- **Acurácia agregada: 65.4%** (acaso: 5.0%, 13.1x acima do acaso)
- Média por sinalizador: 65.4% (melhor: Sinalizador01 81%; pior: Sinalizador10 43%)

## Classes mais bem reconhecidas
- America: 95%
- Espelho: 95%
- Amarelo: 78%
- Barulho: 72%
- Conhecer: 70%

## Classes menos reconhecidas
- Banheiro: 40% (mais confundida com Acontecer)
- Vacina: 50% (mais confundida com Aluno)
- Cinco: 52% (mais confundida com Aproveitar)
- Medo: 52% (mais confundida com Filho)
- Filho: 57% (mais confundida com Aproveitar)

## Principais confusões (real -> previsto)
- Ruim -> Maca: 13x (32.5% da classe)
- Vacina -> Aluno: 11x (27.5% da classe)
- Aluno -> Vacina: 10x (25.0% da classe)
- Vontade -> Conhecer: 9x (22.5% da classe)
- Sapo -> Esquina: 7x (17.5% da classe)
