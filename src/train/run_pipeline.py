"""Bloque 7/8 — Ejecución del pipeline completo

Orquesta la ejecución reproducible del proyecto llamando a los módulos con
`python -m ...` (entrenamiento, evaluación, comparación, barrido de
hiperparámetros, robustez ante ruido y generación de tablas/figuras).

Ejecución:
- `python -m src.train.run_pipeline [--quick] [--skip-hyperparams] [--skip-noise]`

Nota: se espera ejecutar desde la raíz del proyecto (donde existe `data/raw/`).
"""

import argparse
import os
import subprocess
import sys


def run_module(module: str, args: list[str] | None = None) -> None:
    """Ejecuta un módulo con `python -m` y detiene el pipeline si falla."""
    cmd = [sys.executable, "-m", module]
    if args:
        cmd.extend(args)

    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_project_root() -> None:
    """Comprueba que existen los ficheros `.pat` requeridos en `data/raw/`."""
    required_paths = [
        os.path.join("data", "raw", "letterstrain.pat"),
        os.path.join("data", "raw", "lettersval.pat"),
    ]
    missing = [p for p in required_paths if not os.path.exists(p)]
    if missing:
        missing_str = "\n".join(f"- {p}" for p in missing)
        raise FileNotFoundError(
            "No se han encontrado ficheros requeridos. Ejecuta este comando desde la raíz del proyecto.\n"
            f"Faltan:\n{missing_str}"
        )


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos de línea de comandos del pipeline."""
    p = argparse.ArgumentParser(
        description=(
            "Bloque 7 — Pipeline reproducible: entrenamiento, evaluación, comparación, "
            "barrido de hiperparámetros, gráficas/tablas y validación con ruido."
        )
    )

    p.add_argument(
        "--skip-hyperparams",
        action="store_true",
        help="No ejecutar el barrido de hiperparámetros ni sus gráficas.",
    )
    p.add_argument(
        "--skip-noise",
        action="store_true",
        help="No ejecutar la evaluación con ruido (Bloque 6).",
    )
    p.add_argument(
        "--noise-repeats",
        type=int,
        default=100,
        help="Repeticiones por nivel de ruido (se pasa a evaluate_noise).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Modo rápido: reduce repeticiones (útil para probar).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_project_root()

    noise_repeats = int(args.noise_repeats)
    hyper_repeats: int | None = None

    if args.quick:
        noise_repeats = min(noise_repeats, 20)
        hyper_repeats = 2

    # 1) Entrenamiento y gráficos por modelo
    run_module("src.train.train_perceptron")
    run_module("src.viz.plot_history")

    run_module("src.train.train_mlp")
    run_module("src.viz.plot_history_mlp")

    # 2) Evaluación en validación + comparación
    run_module("src.eval.evaluate_perceptron")
    run_module("src.eval.evaluate_mlp")
    run_module("src.eval.compare_models")

    # 3) Hiperparámetros + gráficas
    if not args.skip_hyperparams:
        hyper_args: list[str] = []
        if hyper_repeats is not None:
            hyper_args.extend(["--repeats", str(hyper_repeats)])

        run_module("src.train.run_hyperparams", hyper_args)
        run_module("src.viz.plot_hyperparams")

    # 4) Ruido (robustez)
    if not args.skip_noise:
        run_module("src.eval.evaluate_noise", ["--repeats", str(noise_repeats)])

    # 5) Tablas/gráficas para la memoria
    run_module("src.viz.generate_tables_and_figures")

    print("\n✅ Pipeline terminado. Revisa results/metrics y results/figures.")


if __name__ == "__main__":
    main()
