# Raport - Klasyfikacja Anomalii Erytrocytów z użyciem Sieci Neuronowych

## 1. Architektura Sieci Neuronowej

Projekt wykorzystuje **transfer learning** na bazie przedtrenowanego modelu **ResNet-18** (IMAGENET1K_V1).

### Modyfikacje architektury:

```python
resnet18 = models.resnet18(weights="IMAGENET1K_V1")

# Zamrożenie wszystkich warstw poza ostatnią
for param in resnet18.parameters():
    param.requires_grad = False

# Zastąpienie warstwy klasyfikacyjnej (fully connected)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),      # Redukcja cech do 128
    nn.ReLU(),                     # Aktywacja nieliniowa
    nn.Dropout(0.5),               # Dropout 50% (zapobieganie przeuczeniu)
    nn.Linear(128, num_classes)    # Wyjście dla 2 klas (binarna klasyfikacja)
)
```

**Szczegóły architektury:**
- **Baza:** ResNet-18 z przedtrenowanymi wagami ImageNet
- **Zamrożenie:** Wszystkie warstwy konwolucyjne są zamrożone
- **Warstwa klasyfikacyjna:**
  - `Linear(512 → 128)` - warstwa redukcji wymiarów
  - `ReLU` - funkcja aktywacji
  - `Dropout(p=0.5)` - regularyzacja
  - `Linear(128 → 2)` - warstwa wyjściowa dla 2 klas

---

## 2. Funkcja Straty (Loss Function)

```python
self.loss = nn.CrossEntropyLoss()
```

**CrossEntropyLoss** - standardowa funkcja straty dla problemów klasyfikacji wieloklasowej/binarnej, łącząca Softmax i NLL Loss.

---

## 3. Optimizer

```python
optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
```

**AdamW** (Adam with Weight Decay) - zaawansowana wersja optymalizatora Adam z rozdzieleniem regularyzacji wag.

---

## 4. Hiperparametry

| Hiperparametr | Wartość | Lokalizacja |
|---------------|---------|-------------|
| **Learning Rate** | `1e-3` (0.001) | `LitResNet.__init__(learning_rate=1e-3)` |
| **Batch Size** | `32` | `get_dataloaders(batch_size=32)` |
| **Number of Workers** | `4` | `get_dataloaders(num_workers=4)` |
| **Max Epochs** | `10` | `Trainer(max_epochs=10)` |
| **Dropout Rate** | `0.5` | Warstwa `nn.Dropout(0.5)` |
| **Train/Val/Test Split** | `70% / 15% / 15%` | `random_split` |
| **Random Seed** | `42` | `torch.Generator().manual_seed(42)` |

### Augmentacja danych (Training):
```python
train_transform = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomRotation((0, 0)),      # 0°
        transforms.RandomRotation((90, 90)),    # 90°
        transforms.RandomRotation((180, 180)),  # 180°
        transforms.RandomRotation((270, 270)),  # 270°
    ]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Preprocessing (Validation/Test):
```python
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 5. Metryki Ewaluacji

Model śledzi następujące metryki podczas treningu:

```python
self.metrics = MetricCollection([
    torchmetrics.Accuracy(task="binary"),
    torchmetrics.F1Score(task="binary"),
    torchmetrics.Precision(task="binary"),
    torchmetrics.Recall(task="binary")
])
```

- **Accuracy** - dokładność klasyfikacji
- **F1 Score** - średnia harmoniczna Precision i Recall
- **Precision** - precyzja
- **Recall** - czułość

---

## 6. Podsumowanie Treningu

- **Framework:** PyTorch Lightning
- **Akcelerator:** GPU
- **Logger:** Weights & Biases (WandB)
- **Checkpoint:** Zapisywanie najlepszego modelu na podstawie `val_loss` (minimum)
- **Experiment Name:** `"resnet18_transfer_learning"`
- **Run Name:** `"basic_finetuning"`

---

## 7. Dataset

Dataset składa się z około 22,000 obrazów pojedynczych komórek krwi:
- **Klasa pozytywna:** Komórki zainfekowane patogenem
- **Klasa negatywna:** Zdrowe erytrocyty
- **Źródło:** Segmentowane obrazy z preparatów krwi barwionych metodą Giemsy

---

*Raport wygenerowany automatycznie na podstawie analizy kodu źródłowego projektu.*