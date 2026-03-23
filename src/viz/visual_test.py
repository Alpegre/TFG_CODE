from src.data.pat_loader import load_pat
from src.data.labels import onehot_to_letter
from src.viz.plot_letter import plot_letter

X, y = load_pat("data/raw/letterstrain.pat")

letter = onehot_to_letter(y[10])
plot_letter(X[10], title=f"Letra: {letter}")