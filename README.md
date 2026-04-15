# TFG — Reconocimiento de letras con redes neuronales (TensorFlow)

## Descripción
Este proyecto implementa en **Python + TensorFlow** un sistema de redes neuronales capaz de aprender y reconocer automáticamente las 26 letras del alfabeto a partir de patrones binarios de baja resolución (matrices 7×5).  
El trabajo reproduce y amplía el flujo de trabajo tradicional de JavaNNS, migrándolo a un entorno moderno y reproducible.

---

## ✅ Bloques completados hasta ahora

### **Bloque 1 — Preparación del proyecto**
- Estructura base del proyecto (datos, código, resultados)
- Ficheros `.pat` en `data/raw/`
- Entorno y dependencias en [requirements.txt](requirements.txt)

**Resumen:** Se organizó el repositorio y el entorno para poder ejecutar el flujo completo (entrenamiento → evaluación → tablas/figuras) de forma reproducible.

### **Bloque 2 — Lectura y visualización**
- Lector de ficheros `.pat` (formato SNNS)
- Conversión a arrays/tensores (`X` con 35 entradas, `y` one-hot con 26 clases)
- Visualización 7×5
- Mapeo automático A–Z
- Scripts clave: [src/data/pat_loader.py](src/data/pat_loader.py), [src/data/labels.py](src/data/labels.py), [src/viz/plot_letter.py](src/viz/plot_letter.py), [src/viz/visual_test.py](src/viz/visual_test.py)

**Resumen:** Se implementó el parser de `.pat` y utilidades para validar visualmente que los patrones se cargan correctamente antes de entrenar.

### **Bloque 3 — Perceptrón simple**
- Modelo 35 → 26 (sin capa oculta)
- Entrenamiento completo
- Evaluación en validación
- Guardado de modelo y métricas
- Scripts clave: [src/models/perceptron.py](src/models/perceptron.py), [src/train/train_perceptron.py](src/train/train_perceptron.py), [src/viz/plot_history.py](src/viz/plot_history.py), [src/eval/evaluate_perceptron.py](src/eval/evaluate_perceptron.py)

**Resumen:** Se entrenó el modelo base, se guardaron los históricos y se evaluó en `lettersval.pat` generando accuracy y matriz de confusión.

### **Bloque 4 — MLP con capa oculta**
- Modelo con 1 capa oculta (ReLU)
- Optimización con Adam
- Entrenamiento completo
- Evaluación y métricas en validación
- Comparación Perceptrón vs MLP
- Scripts clave: [src/models/mlp.py](src/models/mlp.py), [src/train/train_mlp.py](src/train/train_mlp.py), [src/viz/plot_history_mlp.py](src/viz/plot_history_mlp.py), [src/eval/evaluate_mlp.py](src/eval/evaluate_mlp.py), [src/eval/compare_models.py](src/eval/compare_models.py)

**Resumen:** Se añadió un modelo más expresivo (MLP) y se replicó el mismo flujo de entrenamiento/evaluación, generando una comparación directa en validación.

### **Bloque 5 — Experimentos de hiperparámetros (learning rate)**
- Barrido de *learning rate* en Perceptrón y MLP
- Varias repeticiones por configuración (estabilidad)
- Parada temprana por umbral de loss ($d_{max}$)
- Exportación de resultados y resumen estadístico (incluye métricas de validación)
- Scripts clave: [src/train/run_hyperparams.py](src/train/run_hyperparams.py), [src/viz/plot_hyperparams.py](src/viz/plot_hyperparams.py)

**Resumen:** Se automatizó el barrido de learning rate con repeticiones y criterio de parada, guardando resultados en CSV y generando gráficas para apoyar la elección final del Bloque 8.

### **Bloque 6 — Validación con ruido**
- Evaluación en `lettersval.pat` con inversión (0↔1) de 2/4/6 píxeles por patrón
- Repeticiones por nivel de ruido para estimar media y desviación típica
- Exportación de métricas (CSV/JSON) y figuras (curva + matrices de confusión por nivel)
- Script clave: [src/eval/evaluate_noise.py](src/eval/evaluate_noise.py)

**Resumen:** Se cuantificó la degradación de accuracy ante ruido y se identificaron patrones/confusiones más difíciles para comparar robustez entre modelos.

### **Bloque 7 — Automatización (pipeline reproducible)**
- Script “orquestador” para ejecutar entrenamiento → evaluación → comparación → hiperparámetros → ruido → tablas/figuras
- Parámetros configurables para el barrido de hiperparámetros (CLI)
- Script clave: [src/train/run_pipeline.py](src/train/run_pipeline.py)

**Resumen:** Se creó un pipeline reproducible para ejecutar el experimento de principio a fin y regenerar automáticamente modelos, métricas, tablas y figuras.

### **Bloque 8 — Consolidación de resultados**
- Elección del modelo final y parámetros recomendados (con criterio de validación/estabilidad/ruido)
- Preparación de tablas y figuras finales (listas para incluir en la memoria)
- Organización de métricas en formatos CSV/Markdown/LaTeX
- Scripts clave: [src/viz/generate_tables_and_figures.py](src/viz/generate_tables_and_figures.py) (exporta tablas `.md`/`.tex`), [src/train/run_pipeline.py](src/train/run_pipeline.py)

**Objetivo:** dejar un conjunto **final** de resultados (métricas, tablas y figuras) coherente y listo para incorporar en la memoria.

**Qué se hace (criterio de consolidación):**
- **Validación limpia** (`lettersval.pat`) para seleccionar el modelo con mejor rendimiento general.
- **Estabilidad** del entrenamiento en el barrido de hiperparámetros (media/desviación típica).
- **Robustez** ante ruido (2/4/6 píxeles), comparando degradación y casos difíciles.

**Resultado esperado:** tablas y figuras finales exportadas en `results/metrics` y `results/figures` para documentar el TFG.

### **Bloque 9 — Documentación y limpieza**
- Expansión de la explicación de cada bloque en este README.
- Cabecera de propósito en cada script (docstring inicial: `"""Bloque X — ..."""`).
- Ajustes menores para que scripts de prueba se ejecuten de forma segura como módulo (`main()` + `if __name__ == "__main__"`).

**Resumen:** se dejó el repositorio más “presentable” para la memoria/defensa: más contexto en README y scripts autoexplicativos.

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

### **Bloque 5 — Barrido de hiperparámetros (learning rate)**
Ejecuta múltiples entrenamientos para **Perceptrón** y **MLP** variando el *learning rate*.

```bash
python -m src.train.run_hyperparams
```

Parámetros por defecto:
- `--learning-rates 0.2 2.0 5.0 10.0`
- `--repeats 5` (repeticiones por valor)
- `--dmax 0.1` (parada temprana cuando `loss <= dmax`)
- `--val-path data/raw/lettersval.pat` (se calcula también `val_loss` y `val_accuracy` para consolidar en Bloque 8)

### **Bloque 5 — Gráficas de hiperparámetros**
Genera gráficas a partir de `results/metrics/hyperparam_summary.csv` (requiere ejecutar el barrido antes).

```bash
python -m src.viz.plot_hyperparams
```

---

### **Bloque 5 — Tablas y gráficas para la memoria**
Genera tablas (`.csv`, `.md`, `.tex`) y gráficas comparativas a partir de los resultados ya guardados.

```bash
python -m src.viz.generate_tables_and_figures
```

---

### **Bloque 6 — Validación con ruido (2/4/6 píxeles)**
Evalúa los modelos guardados sobre `lettersval.pat` inyectando ruido (inversión 0↔1 de **N píxeles** por patrón) y genera métricas/figuras de robustez.

```bash
python -m src.eval.evaluate_noise
```

---

### **Bloque 7 — Pipeline reproducible**
Ejecuta el flujo completo (entrenar, evaluar, comparar, hiperparámetros, tablas/gráficas y ruido):

```bash
python -m src.train.run_pipeline
```

Opcional (modo rápido para probar):

```bash
python -m src.train.run_pipeline --quick
```

Notas:
- El pipeline regenera los ficheros en `results/metrics`, `results/figures` y `results/logs`.
- Además de la tabla comparativa de modelos, se exportan tablas para hiperparámetros y ruido en `.csv`, `.md` y `.tex`.

---

### **Bloque 8 — Consolidación de resultados (final)**
Este bloque consiste en ejecutar el pipeline **sin modo rápido** para dejar un conjunto final de resultados y tablas listo para la memoria.

Comando recomendado (repeticiones suficientes para el ruido):

```bash
python -m src.train.run_pipeline --noise-repeats 100
```

Salida final esperada (lista para incluir en la memoria):
- Comparación en validación: `results/metrics/table_model_comparison.*`
- Resumen de hiperparámetros (incluye validación): `results/metrics/table_hyperparam_summary.*`
- Resumen de robustez con ruido: `results/metrics/table_noise_eval_summary.*`

---

## ✅ Resultados obtenidos (validación)

| Modelo | Accuracy |
|--------|----------|
| Perceptrón | **0.8889** |
| MLP (1 capa oculta, 64) | **1.0000** |

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

### Hiperparámetros (learning rate)
- `results/metrics/hyperparam_results.csv` (todas las ejecuciones)
- `results/metrics/hyperparam_summary.csv` (media/desviación por modelo y LR, incluye métricas en validación)
- `results/metrics/perceptron_lr{LR}_run{N}_history.csv` (histórico por ejecución)
- `results/metrics/mlp_lr{LR}_run{N}_history.csv` (histórico por ejecución)
- `results/figures/hyperparam_loss_vs_lr.png`
- `results/figures/hyperparam_acc_vs_lr.png`
- `results/figures/hyperparam_stability_loss.png`

### Tablas y gráficas para la memoria
- `results/figures/loss_comparison.png`
- `results/figures/accuracy_comparison.png`
- `results/figures/model_comparison_bar.png`
- `results/metrics/table_model_comparison.csv`
- `results/metrics/table_model_comparison.md`
- `results/metrics/table_model_comparison.tex`
- `results/metrics/table_hyperparam_summary.csv`
- `results/metrics/table_hyperparam_summary.md`
- `results/metrics/table_hyperparam_summary.tex`
- `results/metrics/table_noise_eval_summary.csv`
- `results/metrics/table_noise_eval_summary.md`
- `results/metrics/table_noise_eval_summary.tex`

### Validación con ruido
- `results/metrics/noise_eval_runs.csv` (accuracy por repetición)
- `results/metrics/noise_eval_summary.csv` / `results/metrics/noise_eval_summary.json` (resumen)
- `results/metrics/noise_eval_per_sample.csv` (casos difíciles)
- `results/figures/noise_accuracy_vs_pixels.png`
- `results/figures/perceptron_confusion_noise2.png`
- `results/figures/perceptron_confusion_noise4.png`
- `results/figures/perceptron_confusion_noise6.png`
- `results/figures/mlp_confusion_noise2.png`
- `results/figures/mlp_confusion_noise4.png`
- `results/figures/mlp_confusion_noise6.png`

---

## ✅ Notas

- Si aparece un warning de **pandas** sobre `pyarrow`, se puede ignorar o instalar:
```bash
pip install pyarrow
```

- Si al generar tablas (por ejemplo con `python -m src.viz.generate_tables_and_figures`) aparece un error de dependencias opcionales de **pandas**, instala:
```bash
pip install jinja2 tabulate
```

---

## Autor
Alvaro Perez Gregorio