import json
from pathlib import Path
from typing import Dict, Any
import importlib.util
import sys

# Define test cases directly in the grader file
test_cases = {
    "euclidean_distance": {
        "type": "function",
        "datasets": [
            {
                "name": "euclidean_distance_test_1",
                "points": 5,
                "input": {
                    "point1": [0, 0],
                    "point2": [3, 4]
                },
                "expected_output": 5.0
            },
            {
                "name": "euclidean_distance_test_2",
                "points": 5,
                "input": {
                    "point1": [1, 2, 3],
                    "point2": [4, 5, 6]
                },
                "expected_output": 5.196
            },
            {
                "name": "euclidean_distance_test_3",
                "points": 5,
                "input": {
                    "point1": [1, 1],
                    "point2": [1, 1]
                },
                "expected_output": 0.0
            }
        ]
    },
    "naive_bayes_predict": {
        "type": "function",
        "datasets": [
            {
                "name": "naive_bayes_predict_test_1",
                "points": 10,
                "input": {
                    "priors": [0.5, 0.5],
                    "likelihoods": [[0.8, 0.6], [0.4, 0.7]],
                    "features": [1, 0]
                },
                "expected_output": 0
            },
            {
                "name": "naive_bayes_predict_test_2",
                "points": 10,
                "input": {
                    "priors": [0.3, 0.7],
                    "likelihoods": [[0.9, 0.2], [0.3, 0.8]],
                    "features": [1, 1]
                },
                "expected_output": 1
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
                    "data": [[1, 1], [2, 2], [3, 3], [6, 6]],
                    "labels": [0, 0, 1, 1],
                    "query": [4, 4],
                    "k": 3
                },
                "expected_output": 1
            },
            {
                "name": "knn_predict_test_2",
                "points": 10,
                "input": {
                    "data": [[1, 1], [2, 2], [3, 3], [6, 6]],
                    "labels": [0, 0, 1, 1],
                    "query": [2, 2],
                    "k": 1
                },
                "expected_output": 0
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
            'max_score': 0,
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
                    results['max_score'] += points

                    try:
                        if isinstance(input_data, dict):
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
            'max_score': 0,
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