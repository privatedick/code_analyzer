import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any

class Visualization:
    def create_visualizations(self, analysis_results: Dict[str, Any]):
        self._create_complexity_bar_chart(analysis_results)
        self._create_style_issues_pie_chart(analysis_results)
        self._create_function_count_histogram(analysis_results)
        self._create_complexity_heatmap(analysis_results)

    def _create_complexity_bar_chart(self, analysis_results: Dict[str, Any]):
        complexities = [func['complexity'] for file in analysis_results['files'] for func in file['functions']]
        function_names = [f"{file['file']}:{func['name']}" for file in analysis_results['files'] for func in file['functions']]
        
        plt.figure(figsize=(12, 6))
        plt.bar(function_names, complexities)
        plt.title('Function Complexities')
        plt.xlabel('Functions')
        plt.ylabel('Complexity')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('complexity_bar_chart.png')
        plt.close()

    def _create_style_issues_pie_chart(self, analysis_results: Dict[str, Any]):
        issue_counts = {}
        for file in analysis_results['files']:
            for issue_type, issues in file['style_issues'].items():
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + len(issues)
        
        plt.figure(figsize=(10, 10))
        plt.pie(np.array(list(issue_counts.values())), labels=list(issue_counts.keys()), autopct='%1.1f%%')
        plt.title('Style Issues Distribution')
        plt.savefig('style_issues_pie_chart.png')
        plt.close()

    def _create_function_count_histogram(self, analysis_results: Dict[str, Any]):
        function_counts = [len(file['functions']) for file in analysis_results['files']]
        
        plt.figure(figsize=(10, 6))
        plt.hist(function_counts, bins=range(min(function_counts), max(function_counts) + 2, 1))
        plt.title('Distribution of Function Counts per File')
        plt.xlabel('Number of Functions')
        plt.ylabel('Number of Files')
        plt.savefig('function_count_histogram.png')
        plt.close()

    def _create_complexity_heatmap(self, analysis_results: Dict[str, Any]):
        data = []
        for file in analysis_results['files']:
            for func in file['functions']:
                data.append({
                    'file': file['file'],
                    'function': func['name'],
                    'complexity': func['complexity']
                })
        
        df = pd.DataFrame(data)
        pivot_df = df.pivot(index='file', columns='function', values='complexity')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlOrRd', fmt='d')
        plt.title('Complexity Heatmap')
        plt.tight_layout()
        plt.savefig('complexity_heatmap.png')
        plt.close()
