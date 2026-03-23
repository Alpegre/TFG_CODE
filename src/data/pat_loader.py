import re
import numpy as np


def load_pat(path):
    """
    Carga un fichero .pat de SNNS (formato V3.2) y devuelve:
    - X: array (n_patrones, 35)
    - y: array (n_patrones, 26)
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Buscar cabecera con número de patrones/inputs/outputs
    n_patterns = int(re.search(r"(\d+)", [l for l in lines if "No. of patterns" in l][0]).group(1))
    n_inputs = int(re.search(r"(\d+)", [l for l in lines if "No. of input units" in l][0]).group(1))
    n_outputs = int(re.search(r"(\d+)", [l for l in lines if "No. of output units" in l][0]).group(1))

    X, y = [], []
    i = 0
    while i < len(lines):
        if lines[i].startswith("# Input pattern"):
            # Leer 7 filas de 5 valores = 35
            inputs = []
            for j in range(1, 8):
                row = list(map(int, lines[i + j].split()))
                inputs.extend(row)
            X.append(inputs)
            i += 8
        elif lines[i].startswith("# Output pattern"):
            # Leer salida (1 línea con 26 valores)
            outputs = list(map(int, lines[i + 1].split()))
            y.append(outputs)
            i += 2
        else:
            i += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    assert X.shape == (n_patterns, n_inputs), f"X shape {X.shape} != ({n_patterns},{n_inputs})"
    assert y.shape == (n_patterns, n_outputs), f"y shape {y.shape} != ({n_patterns},{n_outputs})"

    return X, y