from collections import defaultdict
import random
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Lesson11Homework:
    @staticmethod
    def stratified_kfold(data, labels, k):
        class_indices = defaultdict(list)
        for i, label in enumerate(labels):
            class_indices[label].append(i)

        folds = [[] for _ in range(k)]
        for indices in class_indices.values():
            for i, index in enumerate(indices):
                folds[i % k].append(index)

        result = []
        for i in range(k):
            test_indices = folds[i]
            train_indices = [idx for j, fold in enumerate(folds) if j != i for idx in fold]
            result.append((train_indices, test_indices))
        return result

    @staticmethod
    def smote(data, labels, target_class):
        target_indices = [i for i, label in enumerate(labels) if label == target_class]
        synthetic_data = []
        while len(synthetic_data) < len(labels) - len(target_indices):
            i, j = random.sample(target_indices, 2)
            new_sample = [(x + y) / 2 for x, y in zip(data[i], data[j])]
            synthetic_data.append(new_sample)

        synthetic_labels = [target_class] * len(synthetic_data)
        return data + synthetic_data, labels + synthetic_labels

    @staticmethod
    def build_pipeline():
        numeric_features = ['age', 'income']
        categorical_features = ['gender']

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])
        return pipeline