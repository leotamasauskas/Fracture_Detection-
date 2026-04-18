# Detecção de Fraturas Ósseas (Bone Fracture Detection) 🦴

Este repositório contém um pipeline completo de Deep Learning focado na segmentação semântica binária de fraturas ósseas utilizando PyTorch. 

## Visão Geral do Projeto

O objetivo principal deste projeto é implementar uma abordagem automatizada para detectar e segmentar regiões fraturadas em imagens de Raio-X. A metodologia adapta anotações de caixas delimitadoras (YOLO format) para máscaras binárias e treina uma Rede Neural Convolucional para realizar a segmentação em nível de pixel.

**Detalhes Técnicos Implementados:**
- **Arquitetura do Modelo:** U-Net com um encoder ResNet34 (pré-treinado no ImageNet).
- **Bibliotecas Usadas:** PyTorch, Segmentation Models PyTorch (SMP), Albumentations, OpenCV.
- **Função de Perda:** Combinação de Binary Cross Entropy (BCE) com Logits Loss e Dice Loss.
- **Pré-processamento:** Aplicação de CLAHE (Contrast Limited Adaptive Histogram Equalization) e Gaussian Blur para realçar estruturas ósseas e atenuar ruídos nas radiografias.
- **Métricas Avaliadas:** IoU (Intersection over Union), Dice Score (F1), Precisão e Recall.

## Pasta dos Dados e Fonte do Dataset

> **Fonte do Dataset:** A base de imagens utilizada é o "Bone Fractures Detection", originário do Mendeley Data, e pode ser acessada aqui: [Dataset Mendeley Data](https://data.mendeley.com/datasets/xwfs6xbk47/1).

Devido ao tamanho das imagens, a base de dados em si não está protegida pelo versionamento diretamente neste repositório. Para reproduzir o projeto, você deve baixar os dados através do link referenciado e organizar a pasta de dados localmente.

### Estrutura da Pasta dos Dados

O código foi desenhado considerando que o dataset esteja na raiz do projeto, dentro de uma pasta chamada exatamente de `Bone Fractures Detection`. O escopo de diretórios deve ser o seguinte:

```text
Bone Fractures Detection/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

- `images/`: Contém as imagens dos raios-X em formato suportado (como `.jpg`).
- `labels/`: Contém os arquivos de texto `.txt` (formato YOLO) com as coordenadas das caixas que delimitam as posições das fraturas.

Ao executar o pipeline, o dataloader customizado faz a conversão automática das *bounding boxes* desse formato YOLO para Máscaras Binárias de segmentação. Todos os raios-x que não possuírem *labels* são diretamente interpretados como ossos saudáveis.

## Como Executar

1. Certifique-se de baixar e configurar corretamente a estrutura da pasta do dataset.
2. Instale as dependências (certifique-se de ter `torch`, `segmentation-models-pytorch`, `opencv-python`, `matplotlib`, `albumentations`, etc).
3. Você pode explorar os experimentos e explicações passo-a-passo executando o Jupyter Notebook:
   ```bash
   jupyter notebook fracture_detection_pipeline.ipynb
   ```
4. Ou, se preferir utilizar o script consolidado de treinamento ponta a ponta, sem interações, rode:
   ```bash
   python run_pipeline.py
   ```

Durante a execução, o script descobre automaticamente os pares de entrada do dataloader, verifica o balanceamento devido à desproporção entre áreas saudáveis e com fratura, salva os pesos do melhor modelo em uma pasta `checkpoints/` em tempo real na progressão das épocas. Avaliações visuais (composição original, máscara e o preenchimento) e gráficos do histórico ficam salvos por padrão na pasta `results/`.
