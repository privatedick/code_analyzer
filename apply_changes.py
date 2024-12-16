# Korrigerad import för Config
try:
    from .config import Config  # Om config.py ligger i samma katalog som apply_changes.py
except ImportError:
    from src.code_analyzer.config import Config  # Om config.py är i src/code_analyzer/

class ApplyChanges:
    def __init__(self, changes):
        """Initialisera ApplyChanges med de förändringar som ska göras."""
        self.changes = changes
        self.config = Config()

    def apply(self):
        """Applicerar alla ändringar."""
        for change in self.changes:
            try:
                self.apply_change(change)
            except Exception as e:
                print(f"Fel vid tillämpning av ändring {change}: {e}")

    def apply_change(self, change):
        """Tillämpa en specifik ändring."""
        # Detta är en placeholder — anpassa till vad en "ändring" innebär i ditt fall
        print(f"Applicerar ändring: {change}")
        # Exempel på hur en ändring kan se ut:
        if change['type'] == 'replace_text':
            self.replace_text(change['file'], change['old'], change['new'])

    def replace_text(self, file_path, old_text, new_text):
        """Byter ut all förekomst av old_text till new_text i filen file_path."""
        try:
            with open(file_path, 'r') as file:
                content = file.read()
            new_content = content.replace(old_text, new_text)
            with open(file_path, 'w') as file:
                file.write(new_content)
            print(f"Text ersatt i {file_path}: '{old_text}' -> '{new_text}'")
        except Exception as e:
            print(f"Fel vid bearbetning av fil {file_path}: {e}")
