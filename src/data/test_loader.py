from pat_loader import load_pat

X, y = load_pat("data/raw/letterstrain.pat")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Primer patrón (entrada):", X[0])
print("Primer patrón (salida):", y[0])