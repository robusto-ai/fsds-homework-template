import json
import importlib.util
import sys
from pathlib import Path

# Define test cases directly in the grader file
test_cases = {
    "max_score": 100,
    "euclidean_distance": {
        "type": "function",
        "datasets": [
            {
                "name": "euclidean_distance_test_1",
                "points": 10,
                "input": {
                    "point1": [0, 0],
                    "point2": [3, 4]
                },
                "expected_output": 5.0
            },
            {
                "name": "euclidean_distance_test_2",
                "points": 10,
                "input": {
                    "point1": [1, 2, 3],
                    "point2": [4, 5, 6]
                },
                "expected_output": 5.196152422706632
            }
        ]
    },
    "knn_predict": {
        "type": "function",
        "datasets": [
            {
                "name": "knn_predict_test_1",
                "points": 10,
                "input": {
                    "train_data": [[1, 2], [2, 3], [3, 4]],
                    "train_labels": [0, 1, 0],
                    "test_point": [2, 2],
                    "k": 1
                },
                "expected_output": 0
            },
            {
                "name": "knn_predict_test_2",
                "points": 10,
                "input": {
                    "train_data": [[1, 2], [2, 3], [3, 4]],
                    "train_labels": [0, 1, 0],
                    "test_point": [2, 2],
                    "k": 3
                },
                "expected_output": 0
            }
        ]
    }
}

# Load student submission
submission_path = Path('../assignments/homework.py')
spec = importlib.util.spec_from_file_location("homework", submission_path)
homework = importlib.util.module_from_spec(spec)
sys.modules["homework"] = homework
spec.loader.exec_module(homework)

def grade_assignment():
    results = {
        'total_score': 0,
        'max_score': 0,
        'feedback': []
    }

    for function_name, test_case in test_cases.items():
        func = getattr(homework, function_name)
        for dataset in test_case['datasets']:
            input_data = dataset['input']
            expected_output = dataset['expected_output']
            points = dataset['points']
            results['max_score'] += points

            try:
                if isinstance(input_data, dict):
                    output = func(**input_data)
                else:
                    output = func(*input_data)

                assert output == expected_output, f"Expected {expected_output}, got {output}"
                results['total_score'] += points
                results['feedback'].append({
                    'function': function_name,
                    'status': 'PASS',
                    'points': points,
                    'max_points': points,
                    'message': ''
                })
            except Exception as e:
                results['feedback'].append({
                    'function': function_name,
                    'status': 'FAIL',
                    'points': 0,
                    'max_points': points,
                    'message': str(e)
                })

    return results

if __name__ == "__main__":
    results = grade_assignment()
    print(json.dumps(results, indent=2))