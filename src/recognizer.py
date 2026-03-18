from numpy.linalg import norm

def compare_embeddings(emb1, emb2):
    return norm(emb1 - emb2)

def is_same_person(emb1, emb2, threshold=0.7):
    distance = compare_embeddings(emb1, emb2)
    return distance < threshold