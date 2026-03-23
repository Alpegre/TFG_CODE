import string

LETTERS = list(string.ascii_uppercase)

def index_to_letter(index: int) -> str:
    return LETTERS[index]

def onehot_to_letter(onehot) -> str:
    idx = int(onehot.argmax())
    return index_to_letter(idx)