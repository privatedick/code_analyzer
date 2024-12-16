import os
import re

# 1. Katalog där vi letar efter Python-filer
CODE_DIR = 'src/code_analyzer'

# 2. Regex-mönstret för att hitta 'from config import X'
PATTERN = r'^from config import (.+)'

# 3. Funktion för att analysera en fil
def analyze_file(file_path):
    """Analyserar en fil och hittar alla rader som matchar mönstret."""
    results = []  # Håll alla ändringar för denna fil
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        match = re.match(PATTERN, line)
        if match:
            new_line = re.sub(PATTERN, r'from code_analyzer.config import \1', line)
            results.append({
                'file': file_path,
                'line_number': i + 1,
                'original': line.strip(),
                'updated': new_line.strip()
            })
    return results

# 4. Funktion för att analysera hela katalogen
def analyze_directory(directory):
    """Analyserar alla Python-filer i en katalog och returnerar alla ändringar."""
    all_results = []
    for root, _, files in os.walk(directory):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                print(f'  [Analyserar fil] {file_path}')
                results = analyze_file(file_path)
                if results:
                    all_results.extend(results)
    return all_results

# 5. Funktion för att skriva ut och spara rapporten
def create_report(results, report_path='change_report.txt'):
    """Skriver ut och sparar förändringsrapporten."""
    if not results:
        print('\n[Inga ändringar behövs] Inga rader matchade mönstret.\n')
        return

    with open(report_path, 'w', encoding='utf-8') as report_file:
        report_file.write('[Förändringsrapport]\n\n')
        
        file_counts = {}  # För att hålla koll på antal ändringar per fil
        for result in results:
            file_path = result['file']
            line_number = result['line_number']
            original = result['original']
            updated = result['updated']
            
            # Skriv ut information till rapporten
            if file_path not in file_counts:
                report_file.write(f'Fil: {file_path}\n')
                file_counts[file_path] = 0
            
            report_file.write(f'  Rad {line_number}:\n')
            report_file.write(f'    Före:  {original}\n')
            report_file.write(f'    Efter: {updated}\n\n')
            file_counts[file_path] += 1
        
        report_file.write('\n[Sammanfattning]\n')
        total_changes = sum(file_counts.values())
        total_files = len(file_counts)
        report_file.write(f'  Totalt antal filer granskade: {total_files}\n')
        report_file.write(f'  Totalt antal rader som behöver ändras: {total_changes}\n')
        report_file.write(f'\nAlla förändringar finns dokumenterade i {report_path}\n')
        
    print(f'\n[Sammanfattning]')
    print(f'  Totalt antal filer granskade: {len(file_counts)}')
    print(f'  Totalt antal rader som behöver ändras: {total_changes}')
    print(f'\nAlla förändringar finns dokumenterade i {report_path}\n')

# 6. Körning av programmet
if __name__ == '__main__':
    print(f'\n[Startar analys av katalogen] {CODE_DIR}\n')
    results = analyze_directory(CODE_DIR)
    create_report(results)
    print('\n[Klar!]')
