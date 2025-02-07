import json
import importlib.util
import sys
from pathlib import Path

# Load test cases
with open('../assignments/test_cases.json') as f:
    test_cases = json.load(f)

# Load student submission
submission_path = Path('../assignments/homework.py')
spec = importlib.util.spec_from_file_location("homework", submission_path)
homework = importlib.util.module_from_spec(spec)
sys.modules["homework"] = homework
spec.loader.exec_module(homework)

def run_tests():
    results = {
        'total_score': 0,
        'max_score': 0,
        'feedback': []
    }

    for function_name, test_case in test_cases['test_cases'].items():
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
    results = run_tests()
    print(json.dumps(results, indent=2))