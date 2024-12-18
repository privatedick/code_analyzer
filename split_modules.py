import os
import re
import difflib
from pathlib import Path
from loguru import logger
from InquirerPy import prompt
from InquirerPy.base.control import Choice

# Konfigurera loggning med rotation
logger.add("file_split_{time}.log", rotation="10 MB", level="DEBUG")

def write_to_file(file_path, content):
    """Skriver innehållet till den specificerade filen och loggar resultatet."""
    try:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(content)
        logger.info(f"Innehållet har skrivits till {file_path}")
    except FileNotFoundError:
        logger.error(f"Filen {file_path} hittades inte. Kontrollera att sökvägen är korrekt.")
    except PermissionError:
        logger.error(f"Åtkomst nekad vid försök att skriva till {file_path}. Kontrollera behörigheter.")
    except Exception as e:
        logger.error(f"Generellt fel vid skrivning till filen {file_path}: {str(e)}")

def read_file(file_path):
    """Läs innehållet i en fil och returnera en lista med rader."""
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            return infile.readlines()
    except FileNotFoundError:
        logger.error(f"Filen {file_path} hittades inte. Kontrollera att sökvägen är korrekt.")
    except PermissionError:
        logger.error(f"Åtkomst nekad vid försök att läsa {file_path}. Kontrollera behörigheter.")
    except Exception as e:
        logger.error(f"Generellt fel vid läsning av filen {file_path}: {str(e)}")
        return []

def is_import_change_only(original_lines, new_lines):
    """Kontrollera om skillnaderna mellan original och ny version endast är i importer."""
    diff = list(difflib.unified_diff(original_lines, new_lines))
    for line in diff:
        if not re.match(r'^\-?from\s+|^\-?import\s+', line):
            return False
    return True

def prompt_user(diff):
    """Fråga användaren om de vill genomföra ändringarna."""
    for line in diff:
        print(line, end='')

    questions = [
        {
            "type": "list",
            "name": "confirmation",
            "message": "Godkänn dessa ändringar?",
            "choices": [
                Choice("yes"),
                Choice("no"),
                Choice("yes to all"),
                Choice("no to all")
            ],
        }
    ]

    response = prompt(questions)
    return response["confirmation"]

def split_combined_file(combined_file, output_dir, override=False):
    """Dela upp innehållet i en sammanslagen fil till respektive ursprungsfiler."""
    if not os.path.exists(combined_file):
        logger.error(f"Den sammanslagna filen {combined_file} finns inte. Kontrollera filnamn och sökväg.")
        return

    try:
        with open(combined_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        logger.error(f"Filen {combined_file} hittades inte vid försök att öppna. Kontrollera att den existerar.")
        return
    except PermissionError:
        logger.error(f"Åtkomst nekad vid försök att läsa {combined_file}. Kontrollera behörigheter.")
        return
    except Exception as e:
        logger.error(f"Generellt fel vid läsning av den sammanslagna filen {combined_file}: {str(e)}")
        return

    if not lines:
        logger.warning(f"Den sammanslagna filen {combined_file} är tom. Kontrollera att den innehåller data.")
        return

    current_file = None
    current_lines = []
    user_decision = None
    for line in lines:
        start_match = re.match(r"# (.+)", line)
        end_match = re.match(r"# End of (.+)", line)

        if start_match:
            if current_file:
                full_output_path = Path(output_dir, current_file)
                original_lines = read_file(full_output_path)
                if original_lines:
                    diff = list(difflib.unified_diff(original_lines, current_lines))
                    if override or is_import_change_only(original_lines, current_lines):
                        if not override:
                            user_response = user_decision or prompt_user(diff)
                            if user_response == "no":
                                continue
                            elif user_response == "no to all":
                                return
                            elif user_response == "yes to all":
                                user_decision = "yes to all"
                        write_to_file(full_output_path, current_lines)
                    else:
                        logger.warning(f"Skillnaderna i {current_file} är inte enbart i importer och inga ändringar gjordes.")
                else:
                    write_to_file(full_output_path, current_lines)
            current_file = start_match.group(1)
            current_lines = []
        elif end_match:
            if current_file:
                full_output_path = Path(output_dir, current_file)
                original_lines = read_file(full_output_path)
                if original_lines:
                    diff = list(difflib.unified_diff(original_lines, current_lines))
                    if override or is_import_change_only(original_lines, current_lines):
                        if not override:
                            user_response = user_decision or prompt_user(diff)
                            if user_response == "no":
                                continue
                            elif user_response == "no to all":
                                return
                            elif user_response == "yes to all":
                                user_decision = "yes to all"
                        write_to_file(full_output_path, current_lines)
                    else:
                        logger.warning(f"Skillnaderna i {current_file} är inte enbart i importer och inga ändringar gjordes.")
                else:
                    write_to_file(full_output_path, current_lines)
                current_file = None
        else:
            if current_file:
                current_lines.append(line)

    # Kontrollera om det finns ofärdiga filskrivningar
    if current_file:
        full_output_path = Path(output_dir, current_file)
        original_lines = read_file(full_output_path)
        if original_lines:
            diff = list(difflib.unified_diff(original_lines, current_lines))
            if override or is_import_change_only(original_lines, current_lines):
                if not override:
                    user_response = user_decision or prompt_user(diff)
                    if user_response == "no":
                        return
                    elif user_response == "no to all":
                        return
                    elif user_response == "yes to all":
                        user_decision = "yes to all"
                write_to_file(full_output_path, current_lines)
            else:
                logger.warning(f"Skillnaderna i {current_file} är inte enbart i importer och inga ändringar gjordes.")
        else:
            write_to_file(full_output_path, current_lines)
        logger.warning(f"Saknad slutkommentar för filen {current_file}. Filen har ändå skrivits.")

def main():
    combined_file = "all_modules_combined.py"
    output_dir = "."  # Projektroten, men kan ändras till en specifik katalog
    override = "--override" in os.sys.argv

    split_combined_file(combined_file, output_dir, override)
    logger.info(f"Uppdelningen av {combined_file} är klar.")

if __name__ == "__main__":
    main()
