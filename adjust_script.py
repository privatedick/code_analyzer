import os
import re
from pathlib import Path

def adjust_imports(file_path):
    """Läser en fil, justerar importer och skriver tillbaka."""
    file_path = Path(file_path)

    if not file_path.is_file():
        print(f"Varning: {file_path} är inte en fil eller så finns den inte. Hoppar över.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        modified_lines = []
        for line in lines:
            # Justera "from config import"
            line = re.sub(r'from config import ([\w, ]+)',
                          r'from src.code_analyzer.config import \1', line)

             # Justera "from analyzer import"
            line = re.sub(r'from analyzer import ([\w, ]+)',
                          r'from src.code_analyzer.analyzer import \1', line)

             # Justera "from code_analysis import"
            line = re.sub(r'from code_analysis import ([\w, ]+)',
                          r'from src.code_analyzer.code_analysis import \1', line)

            # Justera "from main import"
            line = re.sub(r'from main import ([\w, ]+)',
                          r'from src.code_analyzer.main import \1', line)

            # Justera "from project_comparison import"
            line = re.sub(r'from project_comparison import ([\w, ]+)',
                          r'from src.code_analyzer.project_comparison import \1', line)

           # Justera "from visualization import"
            line = re.sub(r'from visualization import ([\w, ]+)',
                          r'from src.code_analyzer.visualization import \1', line)

            modified_lines.append(line)

        with open(file_path, 'w', encoding='utf-8') as f:
             f.writelines(modified_lines)

        print(f"Justeringar genomförda i: {file_path}")

    except Exception as e:
        print(f"Fel vid hantering av {file_path}: {str(e)}")

# Huvudfunktion för att applicera alla ändringar
def main():
    # Lista alla filer som ska ändras
    files_to_adjust = [
        "src/code_analyzer/code_analysis.py",
        "src/code_analyzer/main.py",
        "src/code_analyzer/project_comparison.py",
        "tests/test_analyzer.py",
        "tests/test_code_analysis.py",
        "tests/test_config.py",
        "tests/test_main.py",
        "tests/test_project_comparison.py",
        "tests/test_visualization.py",
    ]

    # Anpassa importer i varje fil
    for file_path in files_to_adjust:
        adjust_imports(file_path)

if __name__ == "__main__":
    main()
