"""
Pipeline U-Net para Detecção de Fraturas Ósseas
Segmentação Semântica Binária com Deep Learning (PyTorch)

Dataset: Bone Fractures Detection (Roboflow - YOLOv8 format)
Modelo: U-Net + ResNet34 (ImageNet)
Abordagem: Labels YOLO (bounding boxes) → máscaras binárias de segmentação
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

# =============================================
# REPRODUCIBILIDADE
# =============================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("PIPELINE U-NET PARA DETECÇÃO DE FRATURAS ÓSSEAS")
print("=" * 60)
print(f'PyTorch version: {torch.__version__}')
print(f'SMP version: {smp.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
print()


# =============================================
# CONFIGURAÇÃO
# =============================================
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "Bone Fractures Detection")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VALID_DIR = os.path.join(DATA_DIR, "valid")
    TEST_DIR = os.path.join(DATA_DIR, "test")
    
    IMAGES_SUBDIR = "images"
    LABELS_SUBDIR = "labels"
    
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    # Imagem - otimizado para CPU
    IMG_SIZE = 256
    IN_CHANNELS = 3
    NUM_CLASSES = 1
    
    # Classes do Dataset
    CLASS_NAMES = [
        'Comminuted', 'Greenstick', 'Healthy', 'Linear',
        'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
        'Transverse Displaced', 'Transverse',
    ]
    
    # Todas as classes de fratura (excluindo Healthy=2)
    FRACTURE_CLASSES = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    HEALTHY_CLASS = 2
    
    # Treinamento - otimizado para CPU
    EPOCHS = 50
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Early Stopping
    PATIENCE = 12
    
    # Modelo
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

print(f'[CONFIG]')
print(f'  Device: {cfg.DEVICE}')
print(f'  Image size: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}')
print(f'  Batch size: {cfg.BATCH_SIZE}')
print(f'  Max epochs: {cfg.EPOCHS}')
print(f'  Early stopping patience: {cfg.PATIENCE}')
print(f'  Fracture classes: {cfg.FRACTURE_CLASSES}')
print(f'  Train dir: {cfg.TRAIN_DIR} (exists: {os.path.exists(cfg.TRAIN_DIR)})')
print(f'  Valid dir: {cfg.VALID_DIR} (exists: {os.path.exists(cfg.VALID_DIR)})')
print(f'  Test dir:  {cfg.TEST_DIR} (exists: {os.path.exists(cfg.TEST_DIR)})')
print()


# =============================================
# PRÉ-PROCESSAMENTO
# =============================================
def apply_clahe(image, clip_limit=3.0, tile_grid_size=(8, 8)):
    """Aplica CLAHE para melhorar contraste em raio-X."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        image = clahe.apply(image)
    return image


def apply_gaussian_blur(image, kernel_size=3):
    """Aplica Gaussian Blur para redução de ruído."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def preprocess_image(image):
    """Pipeline: CLAHE → Gaussian Blur."""
    image = apply_clahe(image, clip_limit=3.0)
    image = apply_gaussian_blur(image, kernel_size=3)
    return image


# =============================================
# CONVERSÃO YOLO → MÁSCARA BINÁRIA
# =============================================
def parse_yolo_label(label_path):
    """Lê arquivo de label YOLO. Retorna lista de (class_id, xc, yc, w, h)."""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                annotations.append((class_id, x_center, y_center, w, h))
    return annotations


def yolo_to_binary_mask(label_path, img_h, img_w, fracture_classes):
    """Converte labels YOLO em máscara de segmentação binária."""
    mask = np.zeros((img_h, img_w), dtype=np.float32)
    annotations = parse_yolo_label(label_path)
    
    for class_id, xc, yc, w, h in annotations:
        if class_id not in fracture_classes:
            continue
        
        x1 = int((xc - w / 2) * img_w)
        y1 = int((yc - h / 2) * img_h)
        x2 = int((xc + w / 2) * img_w)
        y2 = int((yc + h / 2) * img_h)
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        
        mask[y1:y2, x1:x2] = 1.0
    
    return mask


# =============================================
# DESCOBERTA DE PARES IMAGEM/LABEL
# =============================================
def discover_pairs(split_dir, images_subdir='images', labels_subdir='labels'):
    """Descobre pares imagem/label em diretório YOLO."""
    images_dir = os.path.join(split_dir, images_subdir)
    labels_dir = os.path.join(split_dir, labels_subdir)
    
    if not os.path.exists(images_dir):
        print(f'  ERRO: {images_dir} não encontrado')
        return []
    
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    
    label_files = set(os.listdir(labels_dir)) if os.path.exists(labels_dir) else set()
    
    pairs = []
    missing = 0
    
    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = base_name + '.txt'
        
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, label_file)
        
        if label_file in label_files:
            pairs.append((img_path, label_path))
        else:
            pairs.append((img_path, None))
            missing += 1
    
    if missing > 0:
        print(f'  AVISO: {missing} imagens sem label (tratadas como saudáveis)')
    
    return pairs


print("[DADOS] Descobrindo pares...")
train_pairs = discover_pairs(cfg.TRAIN_DIR)
valid_pairs = discover_pairs(cfg.VALID_DIR)
test_pairs = discover_pairs(cfg.TEST_DIR)

print(f'  Train: {len(train_pairs)} pares')
print(f'  Valid: {len(valid_pairs)} pares')
print(f'  Test:  {len(test_pairs)} pares')
print(f'  Total: {len(train_pairs) + len(valid_pairs) + len(test_pairs)} pares')
print()


# =============================================
# ANÁLISE DE DISTRIBUIÇÃO DE CLASSES
# =============================================
print("[ANÁLISE] Distribuição de classes no treino...")
class_counts = {i: 0 for i in range(len(cfg.CLASS_NAMES))}
total_annotations = 0
images_fracture = 0
images_healthy_only = 0

for img_path, label_path in train_pairs:
    if label_path is None:
        images_healthy_only += 1
        continue
    annotations = parse_yolo_label(label_path)
    has_fracture = False
    for class_id, _, _, _, _ in annotations:
        if class_id in class_counts:
            class_counts[class_id] += 1
            total_annotations += 1
        if class_id in cfg.FRACTURE_CLASSES:
            has_fracture = True
    if has_fracture:
        images_fracture += 1
    else:
        images_healthy_only += 1

print(f'  Total de anotações: {total_annotations}')
print(f'  Imagens com fratura: {images_fracture}')
print(f'  Imagens sem fratura: {images_healthy_only}')
for i in sorted(class_counts.keys()):
    pct = class_counts[i] / total_annotations * 100 if total_annotations > 0 else 0
    tipo = 'Saudável' if i == cfg.HEALTHY_CLASS else 'Fratura'
    print(f'    [{i}] {cfg.CLASS_NAMES[i]:25s}: {class_counts[i]:5d} ({pct:5.1f}%) [{tipo}]')

# Gráfico de distribuição
fig, ax = plt.subplots(figsize=(12, 6))
names = [cfg.CLASS_NAMES[i] for i in sorted(class_counts.keys())]
counts = [class_counts[i] for i in sorted(class_counts.keys())]
colors = ['#e74c3c' if i != 2 else '#2ecc71' for i in sorted(class_counts.keys())]

bars = ax.bar(names, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax.set_title('Distribuição de Classes - Dataset de Treino', fontsize=14, fontweight='bold')
ax.set_xlabel('Classe', fontsize=12)
ax.set_ylabel('Quantidade de Anotações', fontsize=12)
ax.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                str(count), ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  Gráfico salvo em: {os.path.join(cfg.RESULTS_DIR, "class_distribution.png")}')
print()


# =============================================
# DATASET CUSTOMIZADO
# =============================================
class BoneFractureDataset(Dataset):
    """Dataset para segmentação de fraturas ósseas a partir de labels YOLO."""
    
    def __init__(self, pairs, fracture_classes, img_size=256,
                 transform=None, apply_preprocess=True):
        self.pairs = pairs
        self.fracture_classes = fracture_classes
        self.img_size = img_size
        self.transform = transform
        self.apply_preprocess = apply_preprocess
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, label_path = self.pairs[idx]
        
        # Carregar imagem (BGR -> RGB)
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f'Imagem não encontrada: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        img_h, img_w = image.shape[:2]
        
        # Gerar máscara binária a partir dos labels YOLO
        if label_path is not None and os.path.exists(label_path):
            binary_mask = yolo_to_binary_mask(label_path, img_h, img_w, self.fracture_classes)
        else:
            binary_mask = np.zeros((img_h, img_w), dtype=np.float32)
        
        # Pré-processamento (CLAHE + Gaussian Blur)
        if self.apply_preprocess:
            image = preprocess_image(image)
        
        # Augmentação e transformação
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
        
        # Garantir dimensão do canal na máscara
        if isinstance(binary_mask, np.ndarray):
            binary_mask = torch.from_numpy(binary_mask).float()
        
        if binary_mask.ndim == 2:
            binary_mask = binary_mask.unsqueeze(0)
        
        return image, binary_mask


# =============================================
# DATA AUGMENTATION
# =============================================
def get_train_transform(img_size):
    """Pipeline de augmentação para treino."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=0,
            p=0.3, border_mode=cv2.BORDER_CONSTANT
        ),
        A.ElasticTransform(alpha=50, sigma=50 * 0.05, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_valid_transform(img_size):
    """Pipeline para validação/teste."""
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


# =============================================
# CRIAÇÃO DOS DATALOADERS
# =============================================
print("[DATASET] Criando datasets e DataLoaders...")

train_dataset = BoneFractureDataset(
    pairs=train_pairs, fracture_classes=cfg.FRACTURE_CLASSES,
    img_size=cfg.IMG_SIZE, transform=get_train_transform(cfg.IMG_SIZE)
)
valid_dataset = BoneFractureDataset(
    pairs=valid_pairs, fracture_classes=cfg.FRACTURE_CLASSES,
    img_size=cfg.IMG_SIZE, transform=get_valid_transform(cfg.IMG_SIZE)
)
test_dataset = BoneFractureDataset(
    pairs=test_pairs, fracture_classes=cfg.FRACTURE_CLASSES,
    img_size=cfg.IMG_SIZE, transform=get_valid_transform(cfg.IMG_SIZE)
)

train_loader = DataLoader(
    train_dataset, batch_size=cfg.BATCH_SIZE,
    shuffle=True, num_workers=0, pin_memory=False, drop_last=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=cfg.BATCH_SIZE,
    shuffle=False, num_workers=0, pin_memory=False
)
test_loader = DataLoader(
    test_dataset, batch_size=1,
    shuffle=False, num_workers=0, pin_memory=False
)

# Verificar shapes
sample_img, sample_mask = next(iter(train_loader))
print(f'  Image batch: {sample_img.shape}  (B, C, H, W)')
print(f'  Mask batch:  {sample_mask.shape}  (B, 1, H, W)')
print(f'  Mask unique: {torch.unique(sample_mask).tolist()}')
print(f'  Train batches: {len(train_loader)}')
print(f'  Valid batches: {len(valid_loader)}')
print(f'  Test batches:  {len(test_loader)}')
print()


# =============================================
# CÁLCULO DO pos_weight
# =============================================
print("[WEIGHT] Calculando pos_weight para balanceamento de classes...")
total_pos = 0
total_neg = 0

for img_path, label_path in tqdm(train_pairs, desc='  Scanning'):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    h, w = img.shape[:2]
    
    if label_path is not None:
        mask = yolo_to_binary_mask(label_path, h, w, cfg.FRACTURE_CLASSES)
    else:
        mask = np.zeros((h, w), dtype=np.float32)
    
    total_pos += mask.sum()
    total_neg += mask.size - mask.sum()

if total_pos == 0:
    print('  AVISO: Nenhum pixel positivo encontrado!')
    pos_weight = torch.tensor([1.0])
else:
    weight = total_neg / total_pos
    weight = min(weight, 50.0)
    pos_weight = torch.tensor([weight]).float()

print(f'  Pixels positivos: {total_pos:,.0f} ({total_pos/(total_pos+total_neg)*100:.2f}%)')
print(f'  Pixels negativos: {total_neg:,.0f} ({total_neg/(total_pos+total_neg)*100:.2f}%)')
print(f'  Razão neg/pos: {total_neg/total_pos:.1f}:1')
print(f'  pos_weight (clamped at 50): {pos_weight.item():.2f}')
print()


# =============================================
# VISUALIZAÇÃO DE AMOSTRAS
# =============================================
print("[VIS] Gerando visualização de amostras...")
fig, axes = plt.subplots(4, 3, figsize=(15, 20))
fracture_pairs = [(ip, lp) for ip, lp in train_pairs if lp is not None]
np.random.seed(SEED)
sample_indices = np.random.choice(len(fracture_pairs), min(4, len(fracture_pairs)), replace=False)

for i, idx in enumerate(sample_indices):
    img_path, label_path = fracture_pairs[idx]
    raw_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    h, w = raw_img.shape[:2]
    preprocessed = preprocess_image(raw_img.copy())
    binary_mask = yolo_to_binary_mask(label_path, h, w, cfg.FRACTURE_CLASSES)
    
    name = os.path.basename(img_path)[:35]
    
    axes[i, 0].imshow(raw_img)
    axes[i, 0].set_title(f'Original\n{name}', fontsize=9)
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(preprocessed)
    axes[i, 1].set_title('CLAHE + GaussBlur', fontsize=9)
    axes[i, 1].axis('off')
    
    overlay = preprocessed.copy().astype(np.float32) / 255.0
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[:, :, 0] = binary_mask  # Vermelho
    blended = np.clip(0.7 * overlay + 0.3 * mask_rgb, 0, 1)
    
    pos_px = int(binary_mask.sum())
    total_px = binary_mask.size
    axes[i, 2].imshow(blended)
    axes[i, 2].set_title(f'Máscara (overlay)\npx positivos: {pos_px} ({pos_px/total_px*100:.2f}%)', fontsize=9)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'sample_data.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  Salvo em: {os.path.join(cfg.RESULTS_DIR, "sample_data.png")}')
print()


# =============================================
# MODELO U-Net + ResNet34
# =============================================
print("[MODELO] Construindo U-Net + ResNet34...")
model = smp.Unet(
    encoder_name=cfg.ENCODER,
    encoder_weights=cfg.ENCODER_WEIGHTS,
    in_channels=cfg.IN_CHANNELS,
    classes=cfg.NUM_CLASSES,
    activation=None
)
model = model.to(cfg.DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  Total de parâmetros: {total_params:,}')
print(f'  Parâmetros treináveis: {trainable_params:,}')
print()


# =============================================
# LOSS FUNCTION: BCE + DICE
# =============================================
class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(mode='binary', from_logits=True)
    
    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


criterion = BCEDiceLoss(pos_weight=pos_weight.to(cfg.DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print("[LOSS] BCE + Dice Loss")
print(f'  pos_weight: {pos_weight.item():.2f}')
print(f'  Optimizer: AdamW (lr={cfg.LEARNING_RATE}, wd={cfg.WEIGHT_DECAY})')
print(f'  Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)')
print()


# =============================================
# MÉTRICAS
# =============================================
def compute_metrics(pred_mask, true_mask, threshold=0.5):
    """Calcula IoU, Dice, Precision, Recall."""
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    intersection = (pred_bin * true_mask).sum()
    union = pred_bin.sum() + true_mask.sum() - intersection
    
    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2 * intersection + 1e-7) / (pred_bin.sum() + true_mask.sum() + 1e-7)
    precision = (intersection + 1e-7) / (pred_bin.sum() + 1e-7)
    recall = (intersection + 1e-7) / (true_mask.sum() + 1e-7)
    
    return {
        'iou': iou.item(), 'dice': dice.item(),
        'precision': precision.item(), 'recall': recall.item()
    }


# =============================================
# TREINAMENTO
# =============================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    metrics_sum = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    n_batches = 0
    
    for images, masks in tqdm(loader, desc='  Train', leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_metrics = compute_metrics(preds.detach(), masks)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_sum = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    n_batches = 0
    
    for images, masks in tqdm(loader, desc='  Valid', leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        preds = model(images)
        loss = criterion(preds, masks)
        
        total_loss += loss.item()
        batch_metrics = compute_metrics(preds, masks)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_metrics = {k: v / n_batches for k, v in metrics_sum.items()}
    return avg_loss, avg_metrics


# =============================================
# LOOP DE TREINAMENTO PRINCIPAL
# =============================================
best_val_loss = float('inf')
patience_counter = 0
history = {
    'train_loss': [], 'val_loss': [],
    'train_dice': [], 'val_dice': [],
    'train_iou': [], 'val_iou': [],
    'lr': []
}
checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth')

print("=" * 60)
print("INICIANDO TREINAMENTO")
print("=" * 60)
print(f'  Device: {cfg.DEVICE}')
print(f'  Epochs: {cfg.EPOCHS}')
print(f'  Patience: {cfg.PATIENCE}')
print(f'  Checkpoint: {checkpoint_path}')
print()

start_time = time.time()

for epoch in range(cfg.EPOCHS):
    epoch_start = time.time()
    print(f'Epoch {epoch+1}/{cfg.EPOCHS}')
    
    # Treinar
    train_loss, train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, cfg.DEVICE
    )
    
    # Validar
    val_loss, val_metrics = validate(
        model, valid_loader, criterion, cfg.DEVICE
    )
    
    # Scheduler
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    # Histórico
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_dice'].append(train_metrics['dice'])
    history['val_dice'].append(val_metrics['dice'])
    history['train_iou'].append(train_metrics['iou'])
    history['val_iou'].append(val_metrics['iou'])
    history['lr'].append(current_lr)
    
    epoch_time = time.time() - epoch_start
    
    print(f'  Train Loss: {train_loss:.4f} | Dice: {train_metrics["dice"]:.4f} | IoU: {train_metrics["iou"]:.4f}')
    print(f'  Valid Loss: {val_loss:.4f} | Dice: {val_metrics["dice"]:.4f} | IoU: {val_metrics["iou"]:.4f}')
    print(f'  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s')
    
    if new_lr < current_lr:
        print(f'  >> LR reduced: {current_lr:.6f} → {new_lr:.6f}')
    
    # Checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        }, checkpoint_path)
        print(f'  ★ Melhor modelo salvo! (val_loss={val_loss:.4f})')
    else:
        patience_counter += 1
        print(f'  Sem melhoria ({patience_counter}/{cfg.PATIENCE})')
    
    # Early Stopping
    if patience_counter >= cfg.PATIENCE:
        print(f'\n⚠ Early stopping na época {epoch+1}')
        break
    
    print()

total_time = time.time() - start_time
print(f'\nTreinamento concluído em {total_time/60:.1f} minutos')
print(f'Melhor val_loss: {best_val_loss:.4f}')
print()


# =============================================
# CURVAS DE TREINAMENTO
# =============================================
print("[VIS] Gerando curvas de treinamento...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_range = range(1, len(history['train_loss']) + 1)

axes[0, 0].plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=1.5)
axes[0, 0].plot(epochs_range, history['val_loss'], 'r-', label='Valid', linewidth=1.5)
axes[0, 0].set_title('Loss (BCE + Dice)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs_range, history['train_dice'], 'b-', label='Train', linewidth=1.5)
axes[0, 1].plot(epochs_range, history['val_dice'], 'r-', label='Valid', linewidth=1.5)
axes[0, 1].set_title('Dice Score (F1)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs_range, history['train_iou'], 'b-', label='Train', linewidth=1.5)
axes[1, 0].plot(epochs_range, history['val_iou'], 'r-', label='Valid', linewidth=1.5)
axes[1, 0].set_title('IoU (Intersection over Union)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(epochs_range, history['lr'], 'g-', linewidth=1.5)
axes[1, 1].set_title('Learning Rate', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  Salvo em: {os.path.join(cfg.RESULTS_DIR, "training_history.png")}')
print()


# =============================================
# AVALIAÇÃO NO TESTE
# =============================================
print("[TESTE] Avaliação no conjunto de teste...")
checkpoint = torch.load(checkpoint_path, map_location=cfg.DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f'  Modelo carregado da época {checkpoint["epoch"]} (val_loss={checkpoint["val_loss"]:.4f})')

test_loss, test_metrics = validate(model, test_loader, criterion, cfg.DEVICE)

print(f'\n  === RESULTADOS NO TESTE ===')
print(f'  Loss:      {test_loss:.4f}')
print(f'  IoU:       {test_metrics["iou"]:.4f}')
print(f'  Dice (F1): {test_metrics["dice"]:.4f}')
print(f'  Precision: {test_metrics["precision"]:.4f}')
print(f'  Recall:    {test_metrics["recall"]:.4f}')
print()


# =============================================
# VISUALIZAÇÃO DAS PREDIÇÕES
# =============================================
print("[VIS] Gerando visualização de predições...")
model.eval()
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

n_vis = min(6, len(test_dataset))
np.random.seed(SEED)
vis_indices = np.random.choice(len(test_dataset), n_vis, replace=False)

fig, axes = plt.subplots(n_vis, 4, figsize=(20, 5*n_vis))
if n_vis == 1:
    axes = axes[np.newaxis, :]

with torch.no_grad():
    for i, idx in enumerate(vis_indices):
        image, mask = test_dataset[idx]
        pred = model(image.unsqueeze(0).to(cfg.DEVICE))
        pred_mask = (torch.sigmoid(pred) > 0.5).float().cpu().squeeze()
        true_mask = mask.squeeze()
        
        img = image.permute(1, 2, 0).numpy()
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        m = compute_metrics(pred.cpu(), mask.unsqueeze(0))
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Imagem', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f'Predição\nDice={m["dice"]:.3f} IoU={m["iou"]:.3f}', fontsize=10)
        axes[i, 2].axis('off')
        
        # Overlay colorido
        overlay = img.copy()
        tp = (pred_mask.numpy() > 0) & (true_mask.numpy() > 0)
        fp = (pred_mask.numpy() > 0) & (true_mask.numpy() == 0)
        fn = (pred_mask.numpy() == 0) & (true_mask.numpy() > 0)
        
        overlay[tp] = [0, 1, 0]
        overlay[fp] = [1, 0, 0]
        overlay[fn] = [0, 0, 1]
        
        blended = np.clip(0.6 * img + 0.4 * overlay, 0, 1)
        
        axes[i, 3].imshow(blended)
        axes[i, 3].set_title('Overlay (V=TP, R=FP, A=FN)', fontsize=10)
        axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'predictions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f'  Salvo em: {os.path.join(cfg.RESULTS_DIR, "predictions.png")}')
print()


# =============================================
# RESUMO FINAL
# =============================================
print("=" * 60)
print("RESUMO FINAL")
print("=" * 60)
print(f'\nDataset: Bone Fractures Detection (Roboflow)')
print(f'  Train: {len(train_pairs)} imagens')
print(f'  Valid: {len(valid_pairs)} imagens')
print(f'  Test:  {len(test_pairs)} imagens')
print(f'\nModelo: U-Net + {cfg.ENCODER} (ImageNet)')
print(f'  Parâmetros treináveis: {trainable_params:,}')
print(f'  Input: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}x3')
print(f'  Output: {cfg.IMG_SIZE}x{cfg.IMG_SIZE}x1 (binário)')
print(f'\nTreinamento:')
print(f'  Épocas treinadas: {len(history["train_loss"])}')
print(f'  Melhor val_loss: {best_val_loss:.4f}')
if len(history['val_dice']) > 0:
    print(f'  Melhor val_dice: {max(history["val_dice"]):.4f}')
    print(f'  Melhor val_iou:  {max(history["val_iou"]):.4f}')
print(f'  Tempo total: {total_time/60:.1f} minutos')
print(f'\nResultados no Teste:')
print(f'  Loss:      {test_loss:.4f}')
print(f'  IoU:       {test_metrics["iou"]:.4f}')
print(f'  Dice (F1): {test_metrics["dice"]:.4f}')
print(f'  Precision: {test_metrics["precision"]:.4f}')
print(f'  Recall:    {test_metrics["recall"]:.4f}')
print(f'\nArquivos salvos:')
print(f'  Checkpoint: {checkpoint_path}')
print(f'  Resultados: {cfg.RESULTS_DIR}/')
for f in os.listdir(cfg.RESULTS_DIR):
    print(f'    - {f}')
print("=" * 60)
