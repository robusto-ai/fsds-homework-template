import json
from pathlib import Path
from typing import Dict, Any
import importlib.util
import sys
import numpy as np

# Define test cases
test_cases = {
    "PredictionRunner": {
        "type": "class",
        "datasets": [
            {
                "name": "runner_init_test",
                "points": 30,
                "input": [np.array([2.0, 3.0])],
                "expected_output": [0]  # Sum = 5.0 < 10.0
            },
            {
                "name": "runner_inference_test",
                "points": 20,
                "input": [np.array([5.0, 6.0])],
                "expected_output": [1]  # Sum = 11.0 > 10.0
            }
        ]
    },
    "create_prediction_service": {
        "type": "function",
        "datasets": [
            {
                "name": "service_creation_test",
                "points": 30,
                "input": [],
                "expected_output": True  # Expecting a valid service
            }
        ]
    },
    "run_simple_inference": {
        "type": "function",
        "datasets": [
            {
                "name": "inference_test_valid",
                "points": 10,
                "input": [np.array([1.0, 2.0])],
                "expected_output": [0]  # Sum = 3.0 < 10.0
            },
            {
                "name": "inference_test_exception",
                "points": 10,
                "input": [None],
                "expected_output": "ValueError raised"
            }
        ]
    }
}

class BentoMLGradingError(Exception):
    pass

def grade_assignment(
    submission_path: Path,
    assignment_config: Dict[str, Any],
    working_dir: Path
) -> Dict[str, Any]:
    try:
        # Import student's submission
        student_file = working_dir / 'homework.py'
        spec = importlib.util.spec_from_file_location("student_submission", student_file)
        student_module = importlib.util.module_from_spec(spec)
        sys.modules["student_submission"] = student_module
        spec.loader.exec_module(student_module)

        results = {
            'total_score': 0,
            'max_score': 0,
            'feedback': [],
            'status': 'COMPLETED'
        }

        # Grade each component
        for component_name, test_case in test_cases.items():
            try:
                if test_case["type"] == "class":
                    runner_class = getattr(student_module, component_name)
                    runner = runner_class()
                else:
                    func = getattr(student_module, component_name)

                function_results = []
                for dataset in test_case['datasets']:
                    input_data = dataset['input']
                    expected_output = dataset['expected_output']
                    points = dataset['points']
                    results['max_score'] += points

                    try:
                        if component_name == "PredictionRunner":
                            output = runner.inference(*input_data)
                        elif component_name == "create_prediction_service":
                            svc = func()
                            output = hasattr(svc, 'runners') and len(svc.runners) > 0 and svc.name == "simple_anomaly_detection"
                        else:  # run_simple_inference
                            if "exception" in dataset["name"]:
                                try:
                                    func(*input_data)
                                    output = "No exception raised"
                                except ValueError:
                                    output = "ValueError raised"
                            else:
                                output = func(*input_data)

                        assert output == expected_output, f"Expected {expected_output}, got {output}"
                        function_results.append({
                            'dataset_name': dataset['name'],
                            'status': 'PASS',
                            'points': points,
                            'max_points': points,
                            'message': ''
                        })
                        results['total_score'] += points
                    except Exception as e:
                        function_results.append({
                            'dataset_name': dataset['name'],
                            'status': 'FAIL',
                            'points': 0,
                            'max_points': points,
                            'message': str(e)
                        })

                results['feedback'].append({
                    'function': component_name,
                    'status': 'PASS' if all(r['status'] == 'PASS' for r in function_results) else 'FAIL',
                    'points': sum(r['points'] for r in function_results),
                    'max_points': sum(r['max_points'] for r in function_results),
                    'datasets': function_results
                })

            except AttributeError:
                results['feedback'].append({
                    'function': component_name,
                    'status': 'MISSING',
                    'points': 0,
                    'message': f'{component_name} not found in submission'
                })
            except Exception as e:
                results['feedback'].append({
                    'function': component_name,
                    'status': 'ERROR',
                    'points': 0,
                    'message': str(e)
                })

        return results
    except Exception as e:
        return {
            'total_score': 0,
            'max_score': 0,
            'feedback': [{'message': str(e)}],
            'status': 'ERROR'
        }

if __name__ == "__main__":
    submission_path = Path('../assignments/homework.py')
    assignment_config = {'test_file': 'grader.py', 'max_score': 100}
    working_dir = submission_path.parent
    results = grade_assignment(submission_path, assignment_config, working_dir)
    print(json.dumps(results, indent=2))