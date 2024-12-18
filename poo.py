import os
import re
from pathlib import Path

# Definiera filerna som ska ändras och deras korrekta import-satser
correct_imports = {
    "src/code_analyzer/analyzer.py": [
        "import logging\n",
        "from typing import Dict, Any\n",
        "from src.code_analyzer.config import Config\n",
        "from src.code_analyzer.code_analysis import CodeAnalysis\n",
        "from src.code_analyzer.project_comparison import ProjectComparison\n",
        "from src.code_analyzer.visualization import Visualization\n",
    ],
    "src/code_analyzer/code_analysis.py": [
        "import os\n",
        "import ast\n",
        "import re\n",
        "import sys\n",
        "import asyncio\n",
        "import logging\n",
        "from concurrent.futures import ThreadPoolExecutor\n",
        "from dataclasses import dataclass\n",
        "from functools import lru_cache\n",
        "from pathlib import Path\n",
        "from typing import Dict, List, Any, Optional\n",
        "from src.code_analyzer.config import Config\n",
    ],
    "src/code_analyzer/config.py": [
        "import os\n",
        "import json\n",
        "import logging\n",
        "from dataclasses import dataclass\n",
        "from enum import Enum\n",
        "from pathlib import Path\n",
        "from typing import Any, Dict, Optional, List, Callable, Union\n",
        "import yaml\n",
        "from cryptography.fernet import Fernet\n",
    ],
    "src/code_analyzer/main.py": [
        "import argparse\n",
        "from src.code_analyzer.config import Config\n",
        "from src.code_analyzer.analyzer import CodeAnalyzer\n",
    ],
    "src/code_analyzer/project_comparison.py": [
        "from typing import Dict, Any, List\n",
        "import difflib\n",
        "from src.code_analyzer.config import Config\n",
        "from src.code_analyzer.code_analysis import CodeAnalysis\n",
    ],
    "src/code_analyzer/visualization.py": [
        "import os\n",
        "import logging\n",
        "from abc import ABC, abstractmethod\n",
        "from dataclasses import dataclass\n",
        "from pathlib import Path\n",
        "from typing import Dict, Any, List, Optional, Union, Tuple\n",
        "from datetime import datetime\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from matplotlib.figure import Figure\n",
        "from matplotlib.axes import Axes\n",
    ],
    "tests/test_analyzer.py": [
        "import pytest\n",
        "import logging\n",
        "from unittest.mock import Mock, patch\n",
        "from src.code_analyzer.config import Config\n",
        "from src.code_analyzer.analyzer import CodeAnalyzer\n",
    ],
    "tests/test_code_analysis.py": [
        "from unittest.mock import Mock, patch\n",
        "import pytest\n",
        "import ast\n",
        "import logging\n",
        "from src.code_analyzer.config import Config\n",
        "from src.code_analyzer.code_analysis import CodeAnalysis\n",
    ],
    "tests/test_config.py": [
        "from unittest.mock import patch, mock_open\n",
        "import pytest\n",
        "import os\n",
        "import json\n",
        "import yaml\n",
        "from src.code_analyzer.config import Config\n",
        "from src.code_analyzer.config import ConfigValue, ConfigValueType\n",
    ],
    "tests/test_main.py": [
        "from unittest.mock import Mock, patch\n",
        "import pytest\n",
        "from unittest.mock import patch, MagicMock\n",
        "from src.code_analyzer.main import main\n",
    ],
    "tests/test_project_comparison.py": [
        "import pytest\n",
        "from unittest.mock import Mock, patch\n",
        "from src.code_analyzer.config import Config\n",
        "from src.code_analyzer.project_comparison import ProjectComparison\n",
    ],
    "tests/test_visualization.py": [
        "from unittest.mock import Mock, patch\n",
        "import pytest\n",
        "from unittest.mock import patch, MagicMock\n",
        "from src.code_analyzer.visualization import Visualization\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import pandas as pd\n",
    ],
}

def adjust_imports(file_path, correct_imports):
    """Skriver över en fils import-satser."""
    file_path = Path(file_path)

    if not file_path.is_file():
        print(f"Varning: {file_path} är inte en fil eller så finns den inte. Hoppar över.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        import_section_start = -1

        # Hitta starten på importsektionen
        for i, line in enumerate(lines):
           if re.match(r'^\s*(?:from|import)\s+', line):
              import_section_start = i
              break

        modified_lines = []
        if import_section_start != -1:
             # Radera gamla import-satser
             for i, line in enumerate(lines):
                if  i >= import_section_start and not re.match(r'^\s*(?:from|import)\s+', line):
                   break
                elif i >= import_section_start:
                    continue
                else:
                    modified_lines.append(line)

             # Lägg till korrekta import-satser
             modified_lines.extend(correct_imports)

             # Lägg till resten av filen
             for i, line in enumerate(lines):
               if i >= import_section_start and not re.match(r'^\s*(?:from|import)\s+', line):
                   modified_lines.append(line)

        else:
              modified_lines = lines
              modified_lines.extend(correct_imports)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(modified_lines)

        print(f"Justeringar genomförda i: {file_path}")

    except Exception as e:
        print(f"Fel vid hantering av {file_path}: {str(e)}")

# Lista över filer som ska fixas
files_to_adjust = [
    "src/code_analyzer/analyzer.py",
    "src/code_analyzer/code_analysis.py",
    "src/code_analyzer/config.py",
    "src/code_analyzer/main.py",
    "src/code_analyzer/project_comparison.py",
    "src/code_analyzer/visualization.py",
    "tests/test_analyzer.py",
    "tests/test_code_analysis.py",
    "tests/test_config.py",
    "tests/test_main.py",
    "tests/test_project_comparison.py",
    "tests/test_visualization.py",
]

# Fixa importer
for file_path in files_to_adjust:
    adjust_imports(file_path, correct_imports[file_path])

print ("importerna har justerats")
