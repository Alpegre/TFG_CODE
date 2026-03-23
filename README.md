# TFG — Reconocimiento de letras con redes neuronales (TensorFlow)

## Descripción
Este proyecto implementa en **Python + TensorFlow** un sistema de redes neuronales capaz de aprender y reconocer automáticamente las 26 letras del alfabeto a partir de patrones binarios de baja resolución (matrices 7×5).  
El trabajo reproduce y amplía el flujo de trabajo tradicional de JavaNNS, migrándolo a un entorno moderno y reproducible.

---

## ✅ Bloques completados hasta ahora

### **Bloque 1 — Preparación del proyecto**
- Estructura base del proyecto
- Ficheros `.pat` en `data/raw/`
- Dependencias en `requirements.txt`

### **Bloque 2 — Lectura y visualización**
- Lector de ficheros `.pat`
- Conversión a tensores/arrays
- Visualización 7×5
- Mapeo automático A–Z

### **Bloque 3 — Perceptrón simple**
- Modelo 35 → 26 (sin capa oculta)
- Entrenamiento completo
- Evaluación en validación
- Guardado de modelo y métricas

### **Bloque 4 — MLP con capa oculta**
- Modelo con 1 capa oculta (ReLU)
- Optimización con Adam
- Entrenamiento completo
- Evaluación y métricas en validación
- Comparación Perceptrón vs MLP

---

## ✅ Estructura del proyecto

```
tfg-alfabeto/
├── data/
│   ├── raw/                 # ficheros .pat originales
│   ├── processed/           # tensores convertidos o npy
├── src/
│   ├── data/                # lector .pat, parsing, labels
│   ├── models/              # modelos TensorFlow
│   ├── train/               # entrenamiento
│   ├── eval/                # validación y métricas
│   ├── viz/                 # gráficos y visualización
├── results/
│   ├── logs/                # modelos guardados
│   ├── figures/             # figuras y gráficos
│   ├── metrics/             # métricas guardadas
├── notebooks/               # (no se usa, solo scripts)
├── README.md
├── requirements.txt
```

---

## ✅ Requisitos

- **Python 3.11**
- TensorFlow 2.15

---

## ✅ Instalación (Windows)

1) **Clonar el repositorio**
```bash
git clone <URL_DEL_REPO>
cd tfg-alfabeto
```

2) **Crear y activar entorno virtual**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

> Si PowerShell bloquea la activación:
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
.venv\Scripts\activate
```

3) **Instalar dependencias**
```bash
pip install -r requirements.txt
```

---

## ✅ Distribución de archivos (IMPORTANTE)

Coloca los ficheros originales en:

```
data/raw/letterstrain.pat
data/raw/lettersval.pat
```

---

## ✅ Ejecución (por bloques)

### **Bloque 2 — Visualización**
```bash
python -m src.viz.visual_test
```

---

### **Bloque 3 — Entrenamiento del perceptrón**
```bash
python -m src.train.train_perceptron
```

---

### **Bloque 3 — Curvas de entrenamiento**
```bash
python -m src.viz.plot_history
```

---

### **Bloque 3 — Evaluación y matriz de confusión**
```bash
python -m src.eval.evaluate_perceptron
```

---

### **Bloque 4 — Entrenamiento del MLP**
```bash
python -m src.train.train_mlp
```

---

### **Bloque 4 — Curvas del MLP**
```bash
python -m src.viz.plot_history_mlp
```

---

### **Bloque 4 — Evaluación del MLP**
```bash
python -m src.eval.evaluate_mlp
```

---

### **Bloque 4 — Comparación Perceptrón vs MLP**
```bash
python -m src.eval.compare_models
```

---

## ✅ Resultados obtenidos (validación)

| Modelo | Accuracy |
|--------|----------|
| Perceptrón | **0.8889** |
| MLP (1 capa oculta, 128) | **1.0000** |

---

## ✅ Resultados generados

### Perceptrón
- `results/logs/perceptron_model.keras`  
- `results/metrics/perceptron_history.csv`  
- `results/metrics/perceptron_eval.json`  
- `results/figures/perceptron_loss.png`  
- `results/figures/perceptron_accuracy.png`  
- `results/figures/perceptron_confusion.png`

### MLP
- `results/logs/mlp_model.keras`  
- `results/metrics/mlp_history.csv`  
- `results/metrics/mlp_eval.json`  
- `results/figures/mlp_loss.png`  
- `results/figures/mlp_accuracy.png`  
- `results/figures/mlp_confusion.png`

### Comparación
- `results/metrics/model_comparison.csv`  
- `results/metrics/model_comparison.json`

---

## ✅ Notas

- Si aparece un warning de **pandas** sobre `pyarrow`, se puede ignorar o instalar:
```bash
pip install pyarrow
```

---

## Autor
Alvaro Perez Gregorio