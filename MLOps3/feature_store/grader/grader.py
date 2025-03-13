import json
from pathlib import Path
from typing import Dict, Any
import importlib.util
import sys
import pandas as pd
from datetime import datetime

# Dummy data for testing
dummy_df = pd.DataFrame({
    "driver_id": ["d1", "d2"],
    "conv_rate": [0.8, 0.6],
    "acc_rate": [0.9, 0.7],
    "event_timestamp": [pd.Timestamp("2025-02-27 10:00"), pd.Timestamp("2025-02-27 10:01")]
})
dummy_df.to_parquet("data/driver_stats.parquet")

# Define test cases
test_cases = {
    "define_driver_features": {
        "type": "function",
        "datasets": [
            {
                "name": "feature_definition_test",
                "points": 30,
                "input": [],
                "expected_output": True  # Valid entity and feature view
            }
        ]
    },
    "materialize_features": {
        "type": "function",
        "datasets": [
            {
                "name": "materialize_test",
                "points": 30,
                "input": [],
                "expected_output": True  # Successful materialization
            }
        ]
    },
    "predict_driver_performance": {
        "type": "function",
        "datasets": [
            {
                "name": "predict_test_d1",
                "points": 20,
                "input": ["d1"],
                "expected_output": {"driver_id": "d1", "conv_rate": 0.8, "acc_rate": 0.9, "performance": "good"}
            },
            {
                "name": "predict_test_invalid",
                "points": 20,
                "input": ["d3"],
                "expected_output": "ValueError raised"
            }
        ]
    }
}

class FeastGradingError(Exception):
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

        # Grade each function
        for function_name, test_case in test_cases.items():
            try:
                func = getattr(student_module, function_name)
                function_results = []
                for dataset in test_case['datasets']:
                    input_data = dataset['input']
                    expected_output = dataset['expected_output']
                    points = dataset['points']
                    results['max_score'] += points

                    try:
                        if function_name == "define_driver_features":
                            entity, view = func()
                            output = (entity.name == "driver" and view.name == "driver_stats" and
                                      len(view.features) == 2)
                            assert output == expected_output, "Invalid feature definitions"
                        elif function_name == "materialize_features":
                            func()
                            output = True
                            assert output == expected_output, "Materialization failed"
                        else:  # predict_driver_performance
                            if "invalid" in dataset["name"]:
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
                    'function': function_name,
                    'status': 'PASS' if all(r['status'] == 'PASS' for r in function_results) else 'FAIL',
                    'points': sum(r['points'] for r in function_results),
                    'max_points': sum(r['max_points'] for r in function_results),
                    'datasets': function_results
                })

            except AttributeError:
                results['feedback'].append({
                    'function': function_name,
                    'status': 'MISSING',
                    'points': 0,
                    'message': f'Function {function_name} not found'
                })
            except Exception as e:
                results['feedback'].append({
                    'function': function_name,
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
    submission_path = Path('homework.py')
    assignment_config = {'test_file': 'grader.py', 'max_score': 100}
    working_dir = Path.cwd()
    working_dir.joinpath("data").mkdir(exist_ok=True)
    dummy_df.to_parquet(working_dir / "data" / "driver_stats.parquet")
    results = grade_assignment(submission_path, assignment_config, working_dir)
    print(json.dumps(results, indent=2))