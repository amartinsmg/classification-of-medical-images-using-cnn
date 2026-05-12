# Análise Comparativa de Arquiteturas de CNNs Pré-Treinadas para Detecção de Pneumonia em Radiografias Torácicas

Para este trabalho, escolhidas as seguintes arquiteturas:

- `"resnet"`: ResNet50
- `"densenet"`: DenseNet121
- `"efficientnet"`: EfficientNetB0

Estas versões foram escolhidas e mantidas ao longo de todos os experimentos por terem complexidade computacional parecida:

- Conseguem rodar num hardware comum com performance parecida, permitindo o treinamento em GPUs domésticas;
- Têm resolução de entrada padrão de 224x224 pixels;
- Todas usam _Global Average Pooling_ (GAP) antes da camada de classificação final, reduzindo drasticamente o número de parâmetros.

Os seguintes parâmetros foram configurados e mantidos ao longo de todos os experimentos:

- Tamanho do lote (_batch_): 32 imagens
- Resolução de entrada das imagens: 224x224 pixels
- Rótulos binários: `0` para normal, `1` para pneumonia
- Embaralhamento do conjunto de treino (apenas)
- Congelamento total das camadas convolucionais do modelo base treinado no conjunto Imagenet
- Otimizador: Adam
- Número de épocas do treino: 10

A construção dos modelos foi feita a partir do seguinte esquema:

![model.keras.svg](./assets/model.keras.svg)

O bloco congelado (f) representa o modelo base pré-treinado no ImageNet, seguido de _pooling_ global e camadas densas para classificação binária. Os shapes do kernel da primeira camada densa refletem a dimensionalidade da saída de cada modelo base: 2048 para ResNet50, 1024 para DenseNet121 e 1280 para EfficientNetB0.

As seguintes características foram alteradas e tiveram seu impacto avaliado ao longo dos experimentos:

- Etapa de normalização
- Etapa de _Data Augmentation_
- Uso do balanceamento de classes
- Taxa de _learning rate_ (taxa de aprendizagem) do otimizador Adam
- Limiar (_threshold_) de decisão do modelo
