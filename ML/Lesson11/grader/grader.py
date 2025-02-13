import json
from pathlib import Path
from typing import Dict, Any
import importlib.util
import sys
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
        student_file = working_dir / 'homework.py'
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

        # Grade each function implementation
        for function_name, test_case in test_cases.items():
            try:
                # Get function from student submission
                func = getattr(student_module, function_name)
                
                # Test all datasets provided for this function
                function_results = []
                for dataset in test_case['datasets']:
                    input_data = dataset['input']
                    expected_output = dataset['expected_output']
                    points = dataset['points']

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
                    'message': f'Function {function_name} not found in submission'
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
            'max_score': assignment_config['max_score'],
            'feedback': [{'message': str(e)}],
            'status': 'ERROR'
        }

if __name__ == "__main__":
    submission_path = Path('../assignments/homework.py')
    assignment_config = {
        'test_file': 'grader.py',
        'max_score': 100
    }
    working_dir = submission_path.parent
    results = grade_assignment(submission_path, assignment_config, working_dir)
    print(json.dumps(results, indent=2))