import json
import numpy as np
from pathlib import Path
from typing import Dict, Any
import importlib.util
import sys
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

test_cases = {
    "LinearRegression": {
        "type": "regression",
        "datasets": [
        {
            "name": "simple_line",
            "points": 20,
            "max_mse": 0.1,
            "X_train": [[1], [2], [3], [4], [5]],
            "y_train": [2, 4, 6, 8, 10],
            "X_test": [[6], [7], [8]],
            "y_test": [12, 14, 16]
        },
        {
            "name": "multiple_features",
            "points": 30,
            "max_mse": 0.5,
            "X_train": [
            [1, 0.5], [2, 1.0], [3, 1.5],
            [4, 2.0], [5, 2.5]
            ],
            "y_train": [3, 6, 9, 12, 15],
            "X_test": [[6, 3.0], [7, 3.5], [8, 4.0]],
            "y_test": [18, 21, 24]
        }
        ]
    },
    "LogisticRegression": {
        "type": "classification",
        "datasets": [
        {
            "name": "binary_classification",
            "points": 25,
            "min_accuracy": 0.9,
            "X_train": [
            [1, 1], [2, 2], [2, 1], [3, 3],
            [1, 2], [4, 4], [5, 5]
            ],
            "y_train": [0, 0, 0, 1, 0, 1, 1],
            "X_test": [[3, 2], [4, 3], [5, 4]],
            "y_test": [1, 1, 1]
        },
        {
            "name": "multifeature_binary",
            "points": 25,
            "min_accuracy": 0.8,
            "X_train": [
            [1, 1, 1], [2, 2, 1], [2, 1, 2],
            [3, 3, 1], [1, 2, 3], [4, 4, 2],
            [5, 5, 1]
            ],
            "y_train": [0, 0, 0, 1, 0, 1, 1],
            "X_test": [[3, 2, 2], [4, 3, 1], [5, 4, 2]],
            "y_test": [1, 1, 1]
        }
        ]
    }
}


class MLGradingError(Exception):
    pass

def grade_assignment(
    submission_path: Path,
    assignment_config: Dict[str, Any],
    working_dir: Path
) -> Dict[str, Any]:
    """
    Grade Machine Learning assignment submission
    
    Args:
        submission_path: Path to directory containing all downloaded files
        assignment_config: Configuration for grading (test_file, max_score)
        working_dir: Path to working directory (same as submission_path in current setup)
    
    Returns:
        Dict containing total_score, max_score, and detailed feedback
    """
    try:
        # Import student's submission from the working directory
        student_file = working_dir / 'implementation.py'
        spec = importlib.util.spec_from_file_location(
            "student_submission",
            student_file
        )
        student_module = importlib.util.module_from_spec(spec)
        sys.modules["student_submission"] = student_module
        spec.loader.exec_module(student_module)
        
        results = {
            'total_score': 0,
            'max_score': assignment_config['max_score'],
            'feedback': [],
            'status': 'COMPLETED'
        }
        
        # Grade each model implementation
        for model_name, test_case in test_cases.items():
            try:
                # Get model class from student submission
                ModelClass = getattr(student_module, model_name)
                
                # Test all datasets provided for this model
                model_results = []
                for dataset in test_case['datasets']:
                    # Load dataset
                    X_train = np.array(dataset['X_train'])
                    y_train = np.array(dataset['y_train'])
                    X_test = np.array(dataset['X_test'])
                    y_test = np.array(dataset['y_test'])
                    
                    # Initialize and train model
                    model = ModelClass()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = {}
                    if test_case['type'] == 'regression':
                        mse = mean_squared_error(y_test, y_pred)
                        metrics['mse'] = mse
                        score = max(0, min(dataset['points'], 
                                        dataset['points'] * (1 - mse/dataset['max_mse'])))
                    else:  # classification
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, average='weighted')
                        recall = recall_score(y_test, y_pred, average='weighted')
                        f1 = f1_score(y_test, y_pred, average='weighted')
                        metrics.update({
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1
                        })
                        score = max(0, min(dataset['points'],
                                       dataset['points'] * (accuracy/dataset['min_accuracy'])))
                    
                    model_results.append({
                        'dataset_name': dataset['name'],
                        'status': 'PASS',
                        'points': score,
                        'max_points': dataset['points'],
                        'metrics': metrics
                    })
                
                # Calculate total score for this model
                model_score = sum(r['points'] for r in model_results)
                results['total_score'] += model_score
                
                results['feedback'].append({
                    'model': model_name,
                    'status': 'PASS',
                    'points': model_score,
                    'max_points': sum(d['points'] for d in test_case['datasets']),
                    'datasets': model_results
                })
                
            except AttributeError:
                results['feedback'].append({
                    'model': model_name,
                    'status': 'MISSING',
                    'points': 0,
                    'message': f'Model class {model_name} not found in submission'
                })
            except Exception as e:
                results['feedback'].append({
                    'model': model_name,
                    'status': 'ERROR',
                    'points': 0,
                    'message': str(e)
                })
        
        return results
    except Exception as e:
        return {
            'total_score': 0,
            'max_score': assignment_config['max_score'],
            'feedback': [{'message': str(e)}],
            'status': 'ERROR'
        }
