from numpy.linalg import norm

#compar faces
def compare_embeddings(emb1, emb2):
    return norm(emb1 - emb2)

#return result if it's same person
def is_same_person(emb1, emb2, threshold=0.9):
    distance = compare_embeddings(emb1, emb2)
    return distance < threshold, distance