#!/usr/bin/env python3
"""
Documentation validation script.

Checks documentation coverage and quality across the codebase.
"""

import os
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Any


class DocumentationValidator:
    """Validator for documentation quality and coverage."""

    def __init__(self):
        self.results = {}
        self.issues = []

    def validate_file(self, filepath: str) -> Dict[str, Any]:
        """Validate documentation in a single file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            tree = ast.parse(content)

            functions = []
            classes = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node)

            # Analyze documentation
            documented_functions = 0
            documented_classes = 0
            quality_issues = []

            for func in functions:
                docstring = ast.get_docstring(func)
                if docstring:
                    documented_functions += 1
                    issues = self._analyze_docstring_quality(docstring, 'function', func.name)
                    quality_issues.extend(issues)
                else:
                    quality_issues.append(f"Function '{func.name}' missing docstring")

            for cls in classes:
                docstring = ast.get_docstring(cls)
                if docstring:
                    documented_classes += 1
                    issues = self._analyze_docstring_quality(docstring, 'class', cls.name)
                    quality_issues.extend(issues)
                else:
                    quality_issues.append(f"Class '{cls.name}' missing docstring")

            total_items = len(functions) + len(classes)
            documented_items = documented_functions + documented_classes
            coverage = documented_items / total_items * 100 if total_items > 0 else 100

            result = {
                'filepath': filepath,
                'total_functions': len(functions),
                'documented_functions': documented_functions,
                'total_classes': len(classes),
                'documented_classes': documented_classes,
                'coverage': coverage,
                'quality_issues': quality_issues
            }

            return result

        except SyntaxError:
            return {
                'filepath': filepath,
                'error': 'Syntax error - cannot parse',
                'coverage': 0,
                'quality_issues': ['File has syntax errors']
            }
        except Exception as e:
            return {
                'filepath': filepath,
                'error': str(e),
                'coverage': 0,
                'quality_issues': [f'Analysis failed: {e}']
            }

    def _analyze_docstring_quality(self, docstring: str, item_type: str, name: str) -> List[str]:
        """Analyze quality of a docstring."""
        issues = []

        if len(docstring.strip()) < 10:
            issues.append(f"{item_type.capitalize()} '{name}': Docstring too short")

        # Check for basic structure
        if item_type == 'function':
            if 'Args:' not in docstring and 'Parameters:' not in docstring:
                # Check if function has parameters
                if '(' in docstring and ')' in docstring:
                    issues.append(f"Function '{name}': Missing Args section")

            if 'Returns:' not in docstring and 'Return:' not in docstring:
                issues.append(f"Function '{name}': Missing Returns section")

        # Check for examples
        if len(docstring) > 200 and 'Example' not in docstring and '>>> ' not in docstring:
            issues.append(f"{item_type.capitalize()} '{name}': Consider adding usage example")

        return issues

    def validate_codebase(self, root_dir: str = "src/python/active_inference") -> Dict[str, Any]:
        """Validate documentation across entire codebase."""
        print("📚 Validating documentation coverage and quality...")

        all_results = []
        total_files = 0
        total_functions = 0
        total_classes = 0
        documented_functions = 0
        documented_classes = 0
        all_quality_issues = []

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    total_files += 1

                    result = self.validate_file(filepath)
                    all_results.append(result)

                    if 'error' not in result:
                        total_functions += result['total_functions']
                        total_classes += result['total_classes']
                        documented_functions += result['documented_functions']
                        documented_classes += result['documented_classes']
                        all_quality_issues.extend(result['quality_issues'])

        # Calculate overall statistics
        overall_coverage = ((documented_functions + documented_classes) /
                          (total_functions + total_classes) * 100) if (total_functions + total_classes) > 0 else 100

        summary = {
            'total_files': total_files,
            'parsable_files': len([r for r in all_results if 'error' not in r]),
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'total_classes': total_classes,
            'documented_classes': documented_classes,
            'overall_coverage': overall_coverage,
            'quality_issues': all_quality_issues,
            'file_results': all_results
        }

        print(f"📊 Documentation Analysis Complete")
        print(f"   Files analyzed: {total_files}")
        print(f"   Parsable files: {summary['parsable_files']}")
        print(f"   Function coverage: {documented_functions}/{total_functions} ({documented_functions/total_functions*100:.1f}%)" if total_functions > 0 else "   Function coverage: N/A")
        print(f"   Class coverage: {documented_classes}/{total_classes} ({documented_classes/total_classes*100:.1f}%)" if total_classes > 0 else "   Class coverage: N/A")
        print(f"   Overall coverage: {overall_coverage:.1f}%")
        print(f"   Quality issues: {len(all_quality_issues)}")

        return summary

    def generate_report(self, summary: Dict[str, Any], output_file: str = None):
        """Generate detailed documentation report."""
        report = "# Documentation Quality Report\n\n"

        report += "## Summary\n\n"
        report += f"- **Total files analyzed:** {summary['total_files']}\n"
        report += f"- **Parsable files:** {summary['parsable_files']}\n"
        report += f"- **Overall documentation coverage:** {summary['overall_coverage']:.1f}%\n"
        report += f"- **Function documentation:** {summary['documented_functions']}/{summary['total_functions']}\n"
        report += f"- **Class documentation:** {summary['documented_classes']}/{summary['total_classes']}\n"
        report += f"- **Quality issues identified:** {len(summary['quality_issues'])}\n\n"

        # Files with lowest coverage
        file_results = [r for r in summary['file_results'] if 'error' not in r]
        sorted_files = sorted(file_results, key=lambda x: x['coverage'])

        report += "## Files Needing Attention\n\n"
        for file_result in sorted_files[:10]:  # Top 10 lowest coverage
            if file_result['coverage'] < 90:
                report += f"### {file_result['filepath']}\n"
                report += f"- Coverage: {file_result['coverage']:.1f}%\n"
                report += f"- Functions: {file_result['documented_functions']}/{file_result['total_functions']}\n"
                report += f"- Classes: {file_result['documented_classes']}/{file_result['total_classes']}\n\n"

        # Quality issues
        if summary['quality_issues']:
            report += "## Quality Issues\n\n"
            for issue in summary['quality_issues'][:20]:  # First 20 issues
                report += f"- {issue}\n"
            report += "\n"

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"📄 Detailed report saved to {output_file}")

        return report


def main():
    """Main documentation validation function."""
    validator = DocumentationValidator()
    summary = validator.validate_codebase()

    # Generate detailed report
    validator.generate_report(summary, "docs/reports/documentation_quality_report.md")

    # Determine success criteria
    min_coverage = 80.0  # Minimum acceptable coverage
    max_issues_per_file = 5  # Maximum quality issues per file

    success = summary['overall_coverage'] >= min_coverage

    if success:
        print("✅ Documentation validation passed")
        sys.exit(0)
    else:
        print(f"❌ Documentation validation failed - coverage {summary['overall_coverage']:.1f}% < {min_coverage}%")
        sys.exit(1)


if __name__ == '__main__':
    main()
