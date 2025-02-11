import json
import importlib.util
import sys
from pathlib import Path

# Define test cases directly in the grader file
test_cases = {
    "max_score": 100,
    "sigmoid": {
        "type": "function",
        "datasets": [
            {
                "name": "sigmoid_test_0",
                "points": 10,
                "input": [0],
                "expected_output": 0.5
            },
            {
                "name": "sigmoid_test_2",
                "points": 10,
                "input": [2],
                "expected_output": 0.8807970779778823
            }
        ]
    },
    "binary_cross_entropy": {
        "type": "function",
        "datasets": [
            {
                "name": "binary_cross_entropy_test",
                "points": 10,
                "input": {
                    "y_true": [1, 0, 1],
                    "y_pred": [0.9, 0.1, 0.8]
                },
                "expected_output": 0.164252033486018
            }
        ]
    },
    "confusion_matrix": {
        "type": "function",
        "datasets": [
            {
                "name": "confusion_matrix_test",
                "points": 10,
                "input": {
                    "y_true": [1, 0, 1, 0],
                    "y_pred": [1, 0, 0, 1]
                },
                "expected_output": {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
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