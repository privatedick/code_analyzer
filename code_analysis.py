import os
import ast
import re
from typing import Dict, Any, List
from config import Config

class CodeAnalysis:
    def __init__(self, config: Config):
        self.config = config

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        project_analysis = {'files': []}
        for root, _, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    project_analysis['files'].append(self.analyze_file(file_path))
        return project_analysis

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r') as file:
            content = file.read()
        
        tree = ast.parse(content)
        
        return {
            'file': file_path,
            'functions': self.extract_functions(tree),
            'complexity': self.calculate_complexity(tree),
            'style_issues': self.check_style(content, tree),
            'imports': self.extract_imports(tree),
        }

    def extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'complexity': self.calculate_complexity(node),
                    'lineno': node.lineno,
                })
        return functions

    def calculate_complexity(self, node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.AsyncFor, ast.AsyncWith)):
                complexity += 1
        return complexity

    def check_style(self, content: str, tree: ast.AST) -> Dict[str, List[str]]:
        issues = {
            'line_length': [],
            'function_names': [],
            'variable_names': [],
            'class_names': [],
            'import_order': [],
        }
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if len(line) > self.config.get('max_line_length', 79):
                issues['line_length'].append(f"Line {i}: Exceeds {self.config.get('max_line_length', 79)} characters")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                issues['function_names'].append(f"Line {node.lineno}: Function name '{node.name}' should use snake_case")
            elif isinstance(node, ast.ClassDef) and not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                issues['class_names'].append(f"Line {node.lineno}: Class name '{node.name}' should use CamelCase")
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.id):
                    issues['variable_names'].append(f"Line {node.lineno}: Variable name '{node.id}' should use snake_case")

        import_lines = [node.lineno for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        if import_lines != sorted(import_lines):
            issues['import_order'].append("Imports are not in ascending order")

        return issues

    def extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"from {node.module} import {', '.join(alias.name for alias in node.names)}")
        return imports
