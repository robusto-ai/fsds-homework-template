import math

# Exercise 1: Implement the Euclidean Distance Function

# Task: Write a Python function euclidean_distance(point1, point2) that calculates the Euclidean distance between two points represented as lists of coordinates.
#
# Example:
#
# ```bash
# >>> euclidean_distance([0, 0], [3, 4])
# 5.0
# >>> euclidean_distance([1, 2, 3], [4, 5, 6])
# 5.196
# >>> euclidean_distance([1, 1], [1, 1])
# 0.0
# ```

def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two points.
    
    Args:
        point1 (list): The first point.
        point2 (list): The second point.
    
    Returns:
        float: The Euclidean distance between the two points.
    """
    pass

# Exercise 2: Implement the Naive Bayes Classifier for Binary Data

# Task: Implement a basic Naive Bayes classifier naive_bayes_predict(priors, likelihoods, features) for binary classification. The function should:
#
# 1. Take the prior probabilities, likelihood probabilities for each feature, and the observed feature values as input.
# 2. Predict the class (0 or 1) based on Bayes' Theorem.
#
# Input Format:
# - priors: A list containing the prior probabilities for class 0 and class 1.
# - likelihoods: A nested list where each sublist contains the likelihood probabilities for each feature in the respective class.
# - features: A list of observed feature values (1 or 0).
#
# Output Format:
# - The predicted class (0 or 1).
#
# Example:
#
# ```bash
# >>> priors = [0.5, 0.5]
# >>> likelihoods = [[0.8, 0.6], [0.4, 0.7]]
# >>> features = [1, 0]
# >>> naive_bayes_predict(priors, likelihoods, features)
# 0
# ```

def naive_bayes_predict(priors, likelihoods, features):
    """
    Predict the class using Naive Bayes classifier for binary data.
    
    Args:
        priors (list): The prior probabilities for class 0 and class 1.
        likelihoods (list of list): The likelihood probabilities for each feature in the respective class.
        features (list): The observed feature values (1 or 0).
    
    Returns:
        int: The predicted class (0 or 1).
    """
    pass

# Exercise 3: Implement K-Nearest Neighbors Classifier

# Task: Implement a function knn_predict(data, labels, query, k) that:
#
# 1. Takes a dataset, corresponding labels, a query point, and the number of neighbors (k) as input.
# 2. Predicts the label for the query point based on the majority class of its k-nearest neighbors.
#
# Input Format:
# - data: A list of points in the dataset, where each point is represented as a list of features.
# - labels: A list of integers representing the class labels for the corresponding points in the dataset.
# - query: A single point represented as a list of features.
# - k: An integer representing the number of nearest neighbors to consider.
#
# Output Format:
# - The predicted class label for the query point (an integer).
#
# Example:
#
# ```bash
# >>> data = [[1, 1], [2, 2], [3, 3], [6, 6]]
# >>> labels = [0, 0, 1, 1]
# >>> query = [4, 4]
# >>> k = 3
# >>> knn_predict(data, labels, query, k)
# 1
# ```

def knn_predict(data, labels, query, k):
    """
    Predict the label of a query point using the k-Nearest Neighbors algorithm.
    
    Args:
        data (list of list): The dataset.
        labels (list): The labels corresponding to the dataset.
        query (list): The query point to predict the label for.
        k (int): The number of neighbors to consider.
    
    Returns:
        int: The predicted label for the query point.
    """
    pass