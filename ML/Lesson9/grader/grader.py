import json
import importlib.util
import sys
from pathlib import Path

# Define test cases directly in the grader file
test_cases = {
    "max_score": 100,
    "bagging_aggregation": {
        "type": "function",
        "datasets": [
            {
                "name": "bagging_aggregation_test_1",
                "points": 10,
                "input": {
                    "predictions": [[1, 0, 1], [1, 1, 1], [0, 0, 1]]
                },
                "expected_output": [1, 0, 1]
            },
            {
                "name": "bagging_aggregation_test_2",
                "points": 10,
                "input": {
                    "predictions": [[1, 0], [0, 1], [1, 0]]
                },
                "expected_output": [1, 0]
            }
        ]
    },
    "update_weights": {
        "type": "function",
        "datasets": [
            {
                "name": "update_weights_test_1",
                "points": 10,
                "input": {
                    "weights": [0.2, 0.3, 0.5],
                    "predictions": [1, 0, 1],
                    "labels": [1, 0, 1]
                },
                "expected_output": [0.2, 0.3, 0.5]
            },
            {
                "name": "update_weights_test_2",
                "points": 10,
                "input": {
                    "weights": [0.2, 0.3, 0.5],
                    "predictions": [1, 0, 1],
                    "labels": [0, 1, 0]
                },
                "expected_output": [0.2, 0.6, 1.0]
            }
        ]
    },
    "stacking_predictions": {
        "type": "function",
        "datasets": [
            {
                "name": "stacking_predictions_test_1",
                "points": 10,
                "input": {
                    "predictions": [[0.8, 0.1], [0.6, 0.4], [0.7, 0.2]],
                    "meta_model": "lambda x: 1 if sum(x) / len(x) > 0.5 else 0"
                },
                "expected_output": [1, 0]
            },
            {
                "name": "stacking_predictions_test_2",
                "points": 10,
                "input": {
                    "predictions": [[1, 0], [1, 1], [0, 0]],
                    "meta_model": "lambda x: 1 if sum(x) > len(x) / 2 else 0"
                },
                "expected_output": [1, 0]
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
                    if 'meta_model' in input_data:
                        meta_model = eval(input_data['meta_model'])
                        input_data['meta_model'] = meta_model
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