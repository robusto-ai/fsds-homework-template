import math

# Exercise 1: Implement the Sigmoid Function

# Task: Write a Python function to calculate the sigmoid function for a given input z. The sigmoid function is defined as:
# 
# sigmoid(z) = 1 / (1 + e^(-z))
#
# Example:
#
# ```bash
# >>> sigmoid(0)
# 0.5
# >>> sigmoid(2)
# 0.8807970779778823
# >>> sigmoid(-2)
# 0.11920292202211755
# ```

def sigmoid(z):
    """
    Compute the sigmoid of z.
    
    Args:
        z (float): The input value.
    
    Returns:
        float: The sigmoid of z.
    """
    pass

# Exercise 2: Compute Binary Cross-Entropy Loss

# Task: Write a Python function to compute the binary cross-entropy loss for a given set of predictions and true labels. Use the formula:
#
# L = -1/N * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))
#
# - Input:
#
#     - y_true: A list of true labels (0 or 1).
#
#     - y_pred: A list of predicted probabilities (values between 0 and 1).
#
# - Output: A single float representing the binary cross-entropy loss.
#
# Example:
#
# ```bash
# >>> binary_cross_entropy([1, 0, 1], [0.9, 0.1, 0.8])
# 0.164252033486018
# ```

def binary_cross_entropy(y_true, y_pred):
    """
    Compute the binary cross-entropy loss.
    
    Args:
        y_true (list): The true labels.
        y_pred (list): The predicted probabilities.
    
    Returns:
        float: The binary cross-entropy loss.
    """
    pass

# Exercise 3: Compute Confusion Matrix

# Task: Write a Python function to compute the confusion matrix for binary classification. The function should return the counts of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
#
# - Input:
#
#     - y_true: A list of true labels (0 or 1).
#
#     - y_pred: A list of predicted labels (0 or 1).
#
# - Output: A dictionary with keys TP, TN, FP, FN.
#
# Example:
#
# ```python
# >>> confusion_matrix([1, 0, 1, 0], [1, 0, 0, 1])
# {'TP': 1, 'TN': 1, 'FP': 1, 'FN': 1}
# ```

def confusion_matrix(y_true, y_pred):
    """
    Compute the confusion matrix.
    
    Args:
        y_true (list): The true labels.
        y_pred (list): The predicted labels.
    
    Returns:
        dict: The confusion matrix with keys 'TP', 'TN', 'FP', 'FN'.
    """
    pass