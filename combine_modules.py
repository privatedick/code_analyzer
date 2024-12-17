import os
from pathlib import Path
from loguru import logger

logger.add("file_{time}.log", rotation="10 MB")  # Lägg till en fil för loggning med rotation

def gather_files(base_dir, extensions):
    """Samlar alla filer med specifika ändelser i en katalog och dess underkataloger."""
    if not os.path.exists(base_dir):
        logger.warning(f"Katalogen {base_dir} finns inte.")
        return []
    try:
        files = [str(file) for ext in extensions for file in Path(base_dir).rglob(f'*.{ext}')]
        logger.info(f"{len(files)} filer hittades i {base_dir}")
        return files
    except Exception as e:
        logger.error(f"Fel vid insamling av filer från {base_dir}: {str(e)}")
        return []

def append_file_content(file_path, outfile):
    """Lägger till innehållet av en fil i en annan fil med markerade kommentarer."""
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            content = infile.read()
            outfile.write(f"# {file_path}\n")
            outfile.write(content)
            outfile.write(f"\n# End of {file_path}\n\n")
        logger.info(f"Innehållet från {file_path} har lagts till.")
    except FileNotFoundError:
        logger.error(f"Filen {file_path} hittades inte. Kontrollera att filen existerar och att sökvägen är korrekt.")
    except PermissionError:
        logger.error(f"Åtkomst nekad vid försök att läsa {file_path}. Kontrollera att du har nödvändiga behörigheter.")
    except Exception as e:
        logger.error(f"Fel vid läsning av {file_path}: {str(e)}")

def main():
    base_dirs = ["src/code_analyzer", "tests"]
    extensions = ["py"]
    special_files = ["pyproject.toml"]
    output_file = "all_modules_combined.py"

    # Kontrollera att utdatafilen inte redan existerar
    if os.path.exists(output_file):
        logger.warning(f"Utdatafilen {output_file} existerar redan. Den kommer att skrivas över.")

    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for base_dir in base_dirs:
                for file_path in gather_files(base_dir, extensions):
                    # Undvik att inkludera själva output-filen
                    if file_path != output_file:
                        append_file_content(file_path, outfile)

            for file_path in special_files:
                if os.path.exists(file_path):
                    append_file_content(file_path, outfile)
                else:
                    logger.warning(f"Specialfilen {file_path} finns inte.")

        logger.info(f"Alla moduler har sammanställts i: {output_file}")
    except PermissionError:
        logger.error(f"Åtkomst nekad vid försök att skriva till {output_file}. Kontrollera att du har nödvändiga behörigheter.")
    except Exception as e:
        logger.error(f"Fel vid skrivning till {output_file}: {str(e)}")

if __name__ == "__main__":
    main()
