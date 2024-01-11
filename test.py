from sklearn.preprocessing import normalize
embedding = [1., 2., 1., 1.]
arr = []
arr.append(embedding)
embedding = normalize(arr, norm='l2')[0]
print(embedding)
a = 0
for i in embedding:
    a += i ** 2
print(a)