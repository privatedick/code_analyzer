import os
import ast
import re
import sys
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Any, Optional
from src.code_analyzer.config import Config

# Anpassad konfigurationsklass
class Config:
    def __init__(self, **kwargs):
        self._config = kwargs

    def get(self, key, default=None):
        return self._config.get(key, default)

# Resultat för analys av filer
@dataclass
class AnalysisResult:
    file_path: Path
    functions: List[Dict[str, Any]]
    complexity: Dict[str, Any]
    style_issues: Dict[str, List[str]]
    imports: Dict[str, Any]
    line_count: int
    error: Optional[str] = None

class CodeAnalysis:
    def __init__(self, config: Config):
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        self._setup_logging()

    def _setup_logging(self) -> None:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(self.config.get('log_level', logging.INFO))

    async def analyze_project(self, project_path: str) -> Dict[str, Any]:
        try:
            project_path = Path(project_path)
            if not project_path.is_dir():
                raise ValueError(f"Invalid project path: {project_path}")

            python_files = list(project_path.rglob("*.py"))
            logger.info(f"Found {len(python_files)} Python files to analyze")

            tasks = [self.analyze_file(file) for file in python_files]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_results = [
                r for r in results
                if isinstance(r, AnalysisResult) and not r.error
            ]
            failed_files = [
                str(python_files[i]) for i, r in enumerate(results)
                if isinstance(r, Exception) or getattr(r, 'error', None)
            ]

            if failed_files:
                logger.warning(f"Failed to analyze {len(failed_files)} files")

            return {
                'results': successful_results,
                'failed_files': failed_files,
                'statistics': self._calculate_project_statistics(successful_results)
            }

        except Exception as e:
            logger.error(f"Project analysis failed: {str(e)}")
            raise

    async def analyze_file(self, file_path: Path) -> AnalysisResult:
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)

            tasks = [
                self._analyze_functions(tree),
                self._analyze_complexity(tree),
                self._analyze_style(content, tree),
                self._analyze_imports(tree)
            ]

            functions, complexity, style_issues, imports = await asyncio.gather(*tasks)

            return AnalysisResult(
                file_path=file_path,
                functions=functions,
                complexity=complexity,
                style_issues=style_issues,
                imports=imports,
                line_count=len(content.splitlines())
            )

        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {str(e)}")
            return AnalysisResult(
                file_path=file_path,
                functions=[],
                complexity={},
                style_issues={},
                imports={},
                line_count=0,
                error=str(e)
            )

    async def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = await self._calculate_node_complexity(node)
                functions.append({
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'returns': self._get_return_annotation(node),
                    'complexity': complexity,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'line_number': node.lineno,
                    'docstring': ast.get_docstring(node),
                    'decorator_list': [ast.unparse(d) for d in node.decorator_list]
                })
        return functions

    @lru_cache(maxsize=128)
    async def _calculate_node_complexity(self, node: ast.AST) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._calculate_complexity_sync,
            node
        )

    def _calculate_complexity_sync(self, node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (
                ast.If, ast.For, ast.While, ast.AsyncFor,
                ast.AsyncWith, ast.Try, ast.ExceptHandler
            )):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    async def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analysera övergripande komplexitetsmetrik."""
        complexity = await self._calculate_node_complexity(tree)
        functions = [
            n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        return {
            'total': complexity,
            'average_per_function': (
                complexity / len(functions) if functions else 0
            ),
            'max_allowed': self.config.get('max_complexity', 10)
        }

    async def _analyze_style(self, content: str, tree: ast.AST) -> Dict[str, List[str]]:
        """Analysera kodstil och konventioner."""
        issues = {
            'line_length': [],
            'naming': [],
            'whitespace': [],
            'docstring': []
        }

        await asyncio.gather(
            self._check_line_issues(content, issues),
            self._check_naming_issues(tree, issues),
            self._check_docstring_issues(tree, issues)
        )

        return issues

    async def _analyze_imports(self, tree: ast.AST) -> Dict[str, Any]:
        """Analysera import-satser och deras organisation."""
        imports = {'standard': [], 'third_party': [], 'local': []}
        import_lines = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    import_type = self._classify_import(name.name)
                    imports[import_type].append({
                        'name': name.name,
                        'alias': name.asname,
                        'line': node.lineno
                    })
                    import_lines.append(node.lineno)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_type = self._classify_import(node.module)
                    imports[import_type].append({
                        'from': node.module,
                        'names': [(n.name, n.asname) for n in node.names],
                        'line': node.lineno
                    })
                    import_lines.append(node.lineno)

        return {
            'imports': imports,
            'ordered': import_lines == sorted(import_lines),
            'total_count': len(import_lines)
        }

    def _classify_import(self, module_name: str) -> str:
        """Klassificera en import som standardbibliotek, tredjeparts eller lokal."""
        stdlib_modules = set(sys.stdlib_module_names)
        first_part = module_name.split('.')[0]
        if first_part in stdlib_modules:
            return 'standard'
        elif '.' in module_name or first_part in {'config', 'utils', 'core'}:
            return 'local'
        return 'third_party'

    async def _check_line_issues(
        self, content: str, issues: Dict[str, List[str]]
    ) -> None:
        """Kontrollera linjer för längd- och whitespaceproblem."""
        max_length = self.config.get('max_line_length', 79)

        for i, line in enumerate(content.splitlines(), 1):
            line = line.rstrip('\n\r')
            if len(line) > max_length:
                issues['line_length'].append(
                    f"Line {i}: Exceeds {max_length} characters"
                )
            if line.endswith(' '):
                issues['whitespace'].append(
                    f"Line {i}: Contains trailing whitespace"
                )

    async def _check_naming_issues(
        self, tree: ast.AST, issues: Dict[str, List[str]]
    ) -> None:
        """Kontrollera namngivningskonventioner."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    issues['naming'].append(
                        f"Line {node.lineno}: Function '{node.name}' "
                        "should use snake_case"
                    )
            elif isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues['naming'].append(
                        f"Line {node.lineno}: Class '{node.name}' "
                        "should use PascalCase"
                    )

    async def _check_docstring_issues(
        self, tree: ast.AST, issues: Dict[str, List[str]]
    ) -> None:
        """Kontrollera docstrings för funktioner och klasser."""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not ast.get_docstring(node):
                    kind = 'Function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'Class'
                    issues['docstring'].append(
                        f"Line {node.lineno}: {kind} '{node.name}' lacks a docstring"
                    )

    def _get_return_annotation(self, node: ast.FunctionDef) -> str:
        """Hämta returtyp-annotering från en funktion."""
        if node.returns:
            return ast.unparse(node.returns)
        return 'Any'

    def _calculate_project_statistics(
        self, results: List[AnalysisResult]
    ) -> Dict[str, Any]:
        """Beräkna övergripande statistik för projektet."""
        if not results:
            return {}

        total_complexity = sum(r.complexity.get('total', 0) for r in results)
        total_functions = sum(len(r.functions) for r in results)
        total_lines = sum(r.line_count for r in results)

        return {
            'file_count': len(results),
            'total_lines': total_lines,
            'total_functions': total_functions,
            'average_file_length': total_lines / len(results),
            'average_complexity': total_complexity / len(results),
            'functions_per_file': total_functions / len(results),
            'style_issues_per_file': sum(
                sum(len(issues) for issues in r.style_issues.values())
                for r in results
            ) / len(results) if results else 0
        }

# Setup för att använda logging korrekt
logger = logging.getLogger(__name__)
