import json
import importlib.util
import sys
from pathlib import Path
import pandas as pd

# Define test cases directly in the grader file
test_cases = {
    "stratified_kfold": {
        "type": "function",
        "datasets": [
            {
                "name": "stratified_kfold_test_1",
                "points": 10,
                "input": {
                    "data": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                    "labels": [0, 1, 0, 1, 0],
                    "k": 2
                },
                "expected_output": [
                    ([2, 4], [0, 1, 3]),
                    ([0, 1, 3], [2, 4])
                ]
            }
        ]
    },
    "smote": {
        "type": "function",
        "datasets": [
            {
                "name": "smote_test_1",
                "points": 10,
                "input": {
                    "data": [[1, 2], [3, 4], [5, 6], [7, 8]],
                    "labels": [0, 1, 0, 1],
                    "target_class": 0
                },
                "expected_output": (
                    [[1, 2], [3, 4], [5, 6], [7, 8], [3, 4]],
                    [0, 1, 0, 1, 0]
                )
            }
        ]
    },
    "build_pipeline": {
        "type": "function",
        "datasets": [
            {
                "name": "build_pipeline_test_1",
                "points": 10,
                "input": {
                    "data": {
                        "age": [25, None, 30],
                        "gender": ["M", "F", "M"],
                        "income": [50000, 60000, None],
                        "target": [1, 0, 1]
                    }
                },
                "expected_output": "pipeline"
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
                    if function_name == "build_pipeline":
                        data = pd.DataFrame(input_data['data'])
                        pipeline = func()
                        pipeline.fit(data[['age', 'gender', 'income']], data['target'])
                        output = "pipeline"
                    else:
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