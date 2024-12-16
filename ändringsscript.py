import os
import re
import shutil

# 1. Definiera sökvägen till katalogen med koden
CODE_DIR = 'src/code_analyzer'

# 2. Definiera det mönster vi söker efter och vad det ska ersättas med
PATTERN = r'^from config import (.+)'  # Regex för att hitta 'from config import X'
REPLACEMENT = r'from code_analyzer.config import \1'  # Nytt mönster med 'code_analyzer.config'

def process_file(file_path):
    """Läser in en fil, gör ändringar och skriver tillbaka den."""
    changes_made = False
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    updated_lines = []
    for i, line in enumerate(lines):
        # Kontrollera om raden matchar mönstret
        if re.match(PATTERN, line):
            new_line = re.sub(PATTERN, REPLACEMENT, line)
            if new_line != line:
                changes_made = True
                print(f'  [Ändrad] {file_path}, rad {i + 1}:')
                print(f'    Före: {line.strip()}')
                print(f'    Efter: {new_line.strip()}')
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)
    
    if changes_made:
        # Säkerhetskopia innan vi skriver över filen
        backup_path = f'{file_path}.bak'
        shutil.copy2(file_path, backup_path)
        print(f'  [Säkerhetskopia skapad] {backup_path}')
        
        # Skriv tillbaka den uppdaterade filen
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(updated_lines)
        print(f'  [Uppdaterad] {file_path}')
    else:
        print(f'  [Ingen ändring behövdes] {file_path}')

def process_directory(directory):
    """Bearbetar alla Python-filer i den angivna katalogen."""
    total_files = 0
    changed_files = 0

    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                print(f'  [Bearbetar fil] {file_path}')
                process_file(file_path)
                total_files += 1
                changed_files += 1
    
    print(f'\n[Sammanfattning]')
    print(f'  Totalt antal filer bearbetade: {total_files}')
    print(f'  Filer med ändringar: {changed_files}')

if __name__ == '__main__':
    print(f'\n[Startar bearbetning av katalogen] {CODE_DIR}\n')
    process_directory(CODE_DIR)
    print('\n[Klar!]')
