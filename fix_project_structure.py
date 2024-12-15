import os
import re

# Konfiguration
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Rotkatalogen för projektet
SRC_DIR = os.path.join(PROJECT_ROOT, 'src', 'code_analyzer')
TESTS_DIR = os.path.join(PROJECT_ROOT, 'tests')
PYPROJECT_FILE = os.path.join(PROJECT_ROOT, 'pyproject.toml')
PYTEST_FILE = os.path.join(PROJECT_ROOT, 'pytest.ini')

# 1. Säkerställ att __init__.py finns i src/code_analyzer och tests
def ensure_init_py_exists():
    for directory in [SRC_DIR, TESTS_DIR]:
        init_path = os.path.join(directory, '__init__.py')
        if not os.path.exists(init_path):
            print(f"Skapar {init_path}")
            with open(init_path, 'w') as f:
                f.write("# Init file för att markera detta som ett paket.\n")

# 2. Uppdatera importvägarna i testfilerna
def update_test_imports():
    for filename in os.listdir(TESTS_DIR):
        if filename.startswith('test_') and filename.endswith('.py'):
            file_path = os.path.join(TESTS_DIR, filename)
            print(f"Uppdaterar imports i {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Uppdatera imports från t.ex. "from config import Config" till "from code_analyzer.config import Config"
            updated_content = re.sub(
                r'from (\w+) import',
                r'from code_analyzer.\1 import',
                content
            )

            # Skriv bara tillbaka filen om något ändrades
            if content != updated_content:
                with open(file_path, 'w') as f:
                    f.write(updated_content)

# 3. Uppdatera pytest.ini för att inkludera src i pythonpath
def update_pytest_ini():
    if not os.path.exists(PYTEST_FILE):
        print(f"Skapar pytest.ini i {PROJECT_ROOT}")
        with open(PYTEST_FILE, 'w') as f:
            f.write("[pytest]\npythonpath = src\n")
    else:
        print(f"Uppdaterar pytest.ini i {PROJECT_ROOT}")
        with open(PYTEST_FILE, 'r') as f:
            content = f.read()
        
        if 'pythonpath = src' not in content:
            with open(PYTEST_FILE, 'a') as f:
                f.write("\n[pytest]\npythonpath = src\n")

# 4. Uppdatera pyproject.toml för att inkludera src/code_analyzer som paket
def update_pyproject_toml():
    if not os.path.exists(PYPROJECT_FILE):
        print(f"pyproject.toml hittades inte i {PROJECT_ROOT}")
        return
    
    with open(PYPROJECT_FILE, 'r') as f:
        content = f.read()
    
    # Lägg till paketkonfiguration om det inte redan finns
    if 'packages = [' not in content:
        print(f"Uppdaterar pyproject.toml för att inkludera src/code_analyzer")
        package_config = """
[tool.poetry]
packages = [
    { include = "code_analyzer", from = "src" }
]
"""
        with open(PYPROJECT_FILE, 'a') as f:
            f.write(package_config)
    else:
        print(f"pyproject.toml verkar redan ha 'packages' konfigurerat")

# 5. Huvudfunktion för att köra alla steg
def main():
    print("🔧 Startar anpassning av projektstrukturen...")
    ensure_init_py_exists()
    update_test_imports()
    update_pytest_ini()
    update_pyproject_toml()
    print("✅ Anpassning klar! Testa att köra:\n\n    poetry install && PYTHONPATH=src poetry run pytest\n")

if __name__ == "__main__":
    main()
