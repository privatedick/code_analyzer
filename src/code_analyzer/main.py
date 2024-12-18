import argparse
from src.code_analyzer.config import Config
from src.code_analyzer.analyzer import CodeAnalyzer

def main():
    # Skapa en parser för kommandoradsargument
    parser = argparse.ArgumentParser(description="Code Analysis Tool")
    parser.add_argument('project_path', type=str, help='Path to the project to analyze')
    parser.add_argument('--compare', type=str, help='Path to another project to compare with', default=None)
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--config', type=str, help='Path to custom config file', default='config.json')
    
    # Parsar argumenten från kommandoraden
    args = parser.parse_args()

    # Ladda konfigurationen
    config = Config(args.config)
    analyzer = CodeAnalyzer(config)

    # Jämför projekt om det anges
    if args.compare:
        results = analyzer.compare_projects(args.project_path, args.compare)
        print("Comparison Results:")
        print(results)
    else:
        # Utför kodanalys om ingen jämförelse är angiven
        results = analyzer.analyze_project(args.project_path)
        print("Analysis Results:")
        print(results)

    # Om visualisering ska genereras
    if args.visualize:
        analyzer.visualize_results(results)
        print("Visualizations have been generated and saved as PNG files.")

# Huvudprogrammet startar här
if __name__ == "__main__":
    main()
