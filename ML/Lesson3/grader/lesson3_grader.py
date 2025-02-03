import json
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any

def grade_assignment(submission_path: Path, assignment_config: Dict[str, Any], working_dir: Path) -> Dict[str, Any]:
    try:
        student_file = working_dir / 'homework.py'
        spec = importlib.util.spec_from_file_location("student_submission", student_file)
        student_module = importlib.util.module_from_spec(spec)
        sys.modules["student_submission"] = student_module
        spec.loader.exec_module(student_module)

        results = {
            'total_score': 0,
            'max_score': assignment_config['max_score'],
            'feedback': [],
            'status': 'COMPLETED'
        }

        test_cases = assignment_config['test_cases']
        for func_name, test_case in test_cases.items():
            try:
                func = getattr(student_module.Lesson3Homework, func_name)
                for case in test_case['cases']:
                    result = func(*case['input'])
                    assert result == case['output'], f"Expected {case['output']}, got {result}"
                    results['total_score'] += case['points']
                    results['feedback'].append({
                        'function': func_name,
                        'status': 'PASS',
                        'points': case['points'],
                        'max_points': case['points'],
                        'input': case['input'],
                        'output': result
                    })
            except AttributeError:
                results['feedback'].append({
                    'function': func_name,
                    'status': 'MISSING',
                    'points': 0,
                    'message': f'Function {func_name} not found in submission'
                })
            except Exception as e:
                results['feedback'].append({
                    'function': func_name,
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