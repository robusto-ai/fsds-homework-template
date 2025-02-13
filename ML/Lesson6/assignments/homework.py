import math

def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions.")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def knn_predict(train_data, train_labels, test_point, k):
    distances = [(euclidean_distance(test_point, point), label) for point, label in zip(train_data, train_labels)]
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return max(set(k_nearest_labels), key=k_nearest_labels.count)