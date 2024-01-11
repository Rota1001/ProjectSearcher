from sklearn.preprocessing import normalize
def normalizor(embedding):
    arr = [embedding]
    return normalize(arr, norm='l2')[0]
