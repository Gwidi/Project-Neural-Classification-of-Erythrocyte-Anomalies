# Raport - Klasyfikacja Anomalii Erytrocytów

Repozytorium: [Project Neural Classification of Erythrocyte Anomalies](https://github.com/Gwidi/Project-Neural-Classification-of-Erythrocyte-Anomalies)

## 1. Architektura Sieci Neuronowej
Projekt oparty na **transfer learning** (ResNet-18, IMAGENET1K_V1). Zamrożone warstwy konwolucyjne, dostosowana warstwa klasyfikacyjna:
- `Linear(512 → 128)` → `ReLU` → `Dropout(0.5)` → `Linear(128 → 2)`

## 2. Funkcja Straty
**CrossEntropyLoss** - łącząca Softmax i NLL Loss.

## 3. Optimizer
**AdamW** - Optymalizator z regularyzacją i lr=1e-3.

## 4. Scheduler
**ReduceLROnPlateau** - Dynamiczne zmniejszanie szybkości uczenia:
- Parametry: factor = 0...