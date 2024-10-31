import argparse
from config import Config
from analyzer import CodeAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Code Analysis Tool")
    parser.add_argument('project_path', type=str, help='Path to the project to analyze')
    parser.add_argument('--compare', type=str, help='Path to another project to compare with', default=None)
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--config', type=str, help='Path to custom config file', default='config.json')

    args = parser.parse_args()

    config = Config(args.config)
    analyzer = CodeAnalyzer(config)

    if args.compare:
        results = analyzer.compare_projects(args.project_path, args.compare)
        print("Comparison Results:")
        print(results)
    else:
        results = analyzer.analyze_project(args.project_path)
        print("Analysis Results:")
        print(results)

    if args.visualize:
        analyzer.visualize_results(results)
        print("Visualizations have been generated and saved as PNG files.")

if __name__ == "__main__":
    main()
