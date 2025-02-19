from collections import Counter

def bagging_aggregation(predictions):
    aggregated = []
    for i in range(len(predictions[0])):
        column = [pred[i] for pred in predictions]
        aggregated.append(Counter(column).most_common(1)[0][0])
    return aggregated

def update_weights(weights, predictions, labels):
    updated_weights = [
        weight * 2 if pred != label else weight
        for weight, pred, label in zip(weights, predictions, labels)
    ]
    total_weight = sum(updated_weights)
    return [w / total_weight for w in updated_weights]

def stacking_predictions(predictions, meta_model):
    transposed = list(zip(*predictions))
    return [meta_model(row) for row in transposed]