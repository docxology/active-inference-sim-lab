#!/usr/bin/env python3
"""
Validation script for unified interfaces.

Checks that all modules properly use unified logging and validation interfaces.
"""

import os
import re
import ast
import sys
from pathlib import Path


class UnifiedInterfaceValidator:
    """Validator for unified interface compliance."""

    def __init__(self):
        self.issues = []
        self.warnings = []

    def validate_file(self, filepath: str) -> bool:
        """Validate a single file for unified interface compliance."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Check for direct logging usage
            if self._has_direct_logging(content):
                self.issues.append(f"{filepath}: Uses direct logging instead of unified logger")

            # Check for direct validation usage
            if self._has_direct_validation(content):
                self.issues.append(f"{filepath}: Uses direct validation instead of unified validator")

            # Check for proper unified interface imports
            if self._needs_unified_interfaces(filepath) and not self._has_unified_imports(content):
                self.warnings.append(f"{filepath}: May benefit from unified interface imports")

            return len([issue for issue in self.issues if filepath in issue]) == 0

        except Exception as e:
            self.issues.append(f"{filepath}: Error during validation - {e}")
            return False

    def _has_direct_logging(self, content: str) -> bool:
        """Check for direct logging usage."""
        # Direct logging imports
        if 'import logging' in content:
            return True

        # Direct logger creation
        if 'logging.getLogger(' in content:
            return True

        # Direct logging method calls (not through unified interface)
        direct_logging_patterns = [
            r'\.info\(',
            r'\.warning\(',
            r'\.error\(',
            r'\.debug\('
        ]

        for pattern in direct_logging_patterns:
            if re.search(pattern, content):
                # Check if it's not using unified logger methods
                if not re.search(r'\.log_(info|warning|error|debug)\(', content):
                    return True

        return False

    def _has_direct_validation(self, content: str) -> bool:
        """Check for direct validation usage."""
        # Direct validation function calls
        validation_functions = [
            'validate_array',
            'validate_matrix',
            'validate_probability_distribution',
            'validate_belief_state',
            'validate_dimensions',
            'validate_hyperparameters'
        ]

        for func in validation_functions:
            if f'{func}(' in content:
                # Check if it's not using unified validator
                if not re.search(r'validator\.' + func, content):
                    return True

        return False

    def _needs_unified_interfaces(self, filepath: str) -> bool:
        """Check if file needs unified interfaces."""
        # Core modules always need unified interfaces
        if any(module in filepath for module in ['core/', 'inference/', 'planning/']):
            return True

        # Files with significant logic need unified interfaces
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Count functions and classes
            tree = ast.parse(content)
            item_count = sum(1 for node in ast.walk(tree)
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef)))

            return item_count > 5  # Files with > 5 functions/classes need unified interfaces

        except:
            return False

    def _has_unified_imports(self, content: str) -> bool:
        """Check for unified interface imports."""
        unified_imports = [
            'from .utils.logging_config import get_unified_logger',
            'from .utils.advanced_validation import get_unified_validator'
        ]

        return any(imp in content for imp in unified_imports)

    def validate_codebase(self, root_dir: str = "src/python/active_inference") -> dict:
        """Validate entire codebase for unified interface compliance."""
        print("🔍 Validating unified interface compliance...")

        total_files = 0
        compliant_files = 0

        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    total_files += 1

                    if self.validate_file(filepath):
                        compliant_files += 1

        compliance_rate = (compliant_files / total_files * 100) if total_files > 0 else 0

        result = {
            'total_files': total_files,
            'compliant_files': compliant_files,
            'compliance_rate': compliance_rate,
            'issues': self.issues,
            'warnings': self.warnings
        }

        print(f"✅ Unified Interface Validation Complete")
        print(f"   Files checked: {total_files}")
        print(f"   Compliant: {compliant_files} ({compliance_rate:.1f}%)")
        print(f"   Issues: {len(self.issues)}")
        print(f"   Warnings: {len(self.warnings)}")

        return result


def main():
    """Main validation function."""
    validator = UnifiedInterfaceValidator()
    result = validator.validate_codebase()

    # Print detailed results
    if result['issues']:
        print("\n🚨 ISSUES FOUND:")
        for issue in result['issues']:
            print(f"   {issue}")

    if result['warnings']:
        print("\n⚠️  WARNINGS:")
        for warning in result['warnings']:
            print(f"   {warning}")

    # Exit with appropriate code
    if result['issues']:
        print(f"\n❌ Validation failed with {len(result['issues'])} issues")
        sys.exit(1)
    else:
        print(f"\n✅ All files compliant with unified interfaces")
        sys.exit(0)


if __name__ == '__main__':
    main()
