import math

class Lesson6Homework:
    @staticmethod
    def euclidean_distance(point1, point2):
        if len(point1) != len(point2):
            raise ValueError("Points must have the same dimensions.")
        if not point1 and not point2:
            return 0
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

    @staticmethod
    def naive_bayes_predict(priors, likelihoods, features):
        if len(likelihoods[0]) != len(features):
            raise ValueError("Number of features must match the likelihoods.")

        posterior_0 = priors[0]
        posterior_1 = priors[1]

        for i, feature in enumerate(features):
            posterior_0 *= likelihoods[0][i] if feature == 1 else (1 - likelihoods[0][i])
            posterior_1 *= likelihoods[1][i] if feature == 1 else (1 - likelihoods[1][i])

        return 0 if posterior_0 > posterior_1 else 1

    @staticmethod
    def knn_predict(data, labels, query, k):
        distances = [(Lesson6Homework.euclidean_distance(query, point), label) for point, label in zip(data, labels)]
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:k]

        class_votes = {}
        for _, label in k_nearest:
            class_votes[label] = class_votes.get(label, 0) + 1

        return max(class_votes, key=class_votes.get)