from typing import Dict, Any, List
import difflib
from src.code_analyzer.config import Config
from src.code_analyzer.code_analysis import CodeAnalysis

class ProjectComparison:
    def __init__(self, config: Config):
        self.config = config
        self.code_analysis = CodeAnalysis(config)

    def compare_projects(self, project1_path: str, project2_path: str) -> Dict[str, Any]:
        project1 = self.code_analysis.analyze_project(project1_path)
        project2 = self.code_analysis.analyze_project(project2_path)

        return {
            'file_count_diff': len(project2['files']) - len(project1['files']),
            'function_count_diff': self._compare_function_counts(project1, project2),
            'complexity_diff': self._compare_complexity(project1, project2),
            'style_issues_diff': self._compare_style_issues(project1, project2),
            'similar_functions': self._find_similar_functions(project1, project2),
            'unique_imports': self._compare_imports(project1, project2),
        }

    def _compare_function_counts(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> int:
        count1 = sum(len(file['functions']) for file in project1['files'])
        count2 = sum(len(file['functions']) for file in project2['files'])
        return count2 - count1

    def _compare_complexity(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> float:
        complexity1 = sum(file['complexity'] for file in project1['files'])
        complexity2 = sum(file['complexity'] for file in project2['files'])
        return (complexity2 - complexity1) / len(project1['files'])

    def _compare_style_issues(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> Dict[str, int]:
        issues1 = self._count_style_issues(project1)
        issues2 = self._count_style_issues(project2)
        return {key: issues2.get(key, 0) - issues1.get(key, 0) for key in set(issues1) | set(issues2)}

    def _count_style_issues(self, project: Dict[str, Any]) -> Dict[str, int]:
        counts = {}
        for file in project['files']:
            for issue_type, issues in file['style_issues'].items():
                counts[issue_type] = counts.get(issue_type, 0) + len(issues)
        return counts

    def _find_similar_functions(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> List[Dict[str, Any]]:
        similar_functions = []
        for file1 in project1['files']:
            for func1 in file1['functions']:
                for file2 in project2['files']:
                    for func2 in file2['functions']:
                        similarity = difflib.SequenceMatcher(None, func1['name'], func2['name']).ratio()
                        if similarity > self.config.get('similarity_threshold', 0.8):
                            similar_functions.append({
                                'function1': f"{file1['file']}:{func1['name']}",
                                'function2': f"{file2['file']}:{func2['name']}",
                                'similarity': similarity
                            })
        return similar_functions

    def _compare_imports(self, project1: Dict[str, Any], project2: Dict[str, Any]) -> Dict[str, List[str]]:
        imports1 = set(import_stmt for file in project1['files'] for import_stmt in file['imports'])
        imports2 = set(import_stmt for file in project2['files'] for import_stmt in file['imports'])
        return {
            'unique_to_project1': list(imports1 - imports2),
            'unique_to_project2': list(imports2 - imports1)
        }
