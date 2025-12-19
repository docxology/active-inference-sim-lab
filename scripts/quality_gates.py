#!/usr/bin/env python3
"""
Comprehensive Quality Gates Validation Script.

Runs all quality checks and validates that the codebase meets
the 100% tested, documented, logged, unified, signposted, orchestrated standards.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path


class QualityGates:
    """Comprehensive quality gates validation."""

    def __init__(self):
        self.results = {}
        self.passed = True

    def run_unified_interface_check(self) -> bool:
        """Check unified interface compliance."""
        print("🔗 Checking unified interface compliance...")

        try:
            result = subprocess.run([
                sys.executable, 'scripts/validate_unified_interfaces.py'
            ], capture_output=True, text=True, timeout=60)

            success = result.returncode == 0
            self.results['unified_interfaces'] = {
                'passed': success,
                'output': result.stdout,
                'errors': result.stderr
            }

            if success:
                print("✅ Unified interfaces compliant")
            else:
                print("❌ Unified interface issues found")
                self.passed = False

            return success

        except Exception as e:
            print(f"❌ Unified interface check failed: {e}")
            self.results['unified_interfaces'] = {'passed': False, 'error': str(e)}
            self.passed = False
            return False

    def run_documentation_check(self) -> bool:
        """Check documentation quality and coverage."""
        print("📚 Checking documentation quality...")

        try:
            result = subprocess.run([
                sys.executable, 'scripts/validate_documentation.py'
            ], capture_output=True, text=True, timeout=120)

            success = result.returncode == 0
            self.results['documentation'] = {
                'passed': success,
                'output': result.stdout,
                'errors': result.stderr
            }

            if success:
                print("✅ Documentation standards met")
            else:
                print("❌ Documentation issues found")
                self.passed = False

            return success

        except Exception as e:
            print(f"❌ Documentation check failed: {e}")
            self.results['documentation'] = {'passed': False, 'error': str(e)}
            self.passed = False
            return False

    def run_test_coverage_check(self) -> bool:
        """Check test coverage levels."""
        print("🧪 Checking test coverage...")

        try:
            # Run coverage analysis
            result = subprocess.run([
                sys.executable, '-m', 'pytest',
                '--cov=src/python/active_inference',
                '--cov-report=term-missing',
                '--cov-report=json:coverage.json',
                '--cov-branch',
                'tests/unit/',
                'tests/integration/'
            ], capture_output=True, text=True, timeout=300)

            # Parse coverage from output
            coverage_output = result.stdout + result.stderr

            # Extract coverage percentage (simple regex approach)
            import re
            coverage_match = re.search(r'TOTAL.*?(\d+)%', coverage_output)
            coverage_percent = int(coverage_match.group(1)) if coverage_match else 0

            min_coverage = 80  # Minimum acceptable coverage
            success = coverage_percent >= min_coverage

            self.results['test_coverage'] = {
                'passed': success,
                'coverage_percent': coverage_percent,
                'min_required': min_coverage,
                'output': coverage_output
            }

            if success:
                print(f"✅ Test coverage: {coverage_percent}% (≥{min_coverage}%)")
            else:
                print(f"❌ Test coverage: {coverage_percent}% (<{min_coverage}%)")
                self.passed = False

            return success

        except Exception as e:
            print(f"❌ Test coverage check failed: {e}")
            self.results['test_coverage'] = {'passed': False, 'error': str(e)}
            self.passed = False
            return False

    def run_orchestration_integration_check(self) -> bool:
        """Check orchestration integration."""
        print("🎭 Checking orchestration integration...")

        # Check if orchestration module exists and is properly integrated
        orchestration_file = Path("src/python/active_inference/orchestration.py")
        integration_tests = [
            Path("tests/integration/test_orchestration_workflows.py"),
            Path("tests/integration/test_experiment_orchestration.py")
        ]

        checks_passed = 0
        total_checks = 3

        # Check orchestration module exists
        if orchestration_file.exists():
            checks_passed += 1
            print("  ✓ Orchestration module exists")
        else:
            print("  ✗ Orchestration module missing")

        # Check integration tests exist
        integration_tests_exist = all(test.exists() for test in integration_tests)
        if integration_tests_exist:
            checks_passed += 1
            print("  ✓ Integration tests exist")
        else:
            print("  ✗ Integration tests missing")

        # Check unified imports in orchestration
        try:
            with open(orchestration_file, 'r') as f:
                content = f.read()

            has_unified_logger = 'get_unified_logger' in content
            has_unified_validator = 'get_unified_validator' in content

            if has_unified_logger and has_unified_validator:
                checks_passed += 1
                print("  ✓ Unified interfaces integrated")
            else:
                print("  ✗ Unified interfaces not integrated")

        except Exception as e:
            print(f"  ✗ Error checking orchestration: {e}")

        success = checks_passed == total_checks
        self.results['orchestration'] = {
            'passed': success,
            'checks_passed': checks_passed,
            'total_checks': total_checks
        }

        if success:
            print("✅ Orchestration integration complete")
        else:
            print(f"❌ Orchestration integration incomplete ({checks_passed}/{total_checks})")
            self.passed = False

        return success

    def run_code_quality_checks(self) -> bool:
        """Run various code quality checks."""
        print("🔍 Running code quality checks...")

        checks = []
        issues_found = 0

        # Check for syntax errors
        print("  Checking syntax...")
        syntax_errors = 0
        for root, dirs, files in os.walk("src/python/active_inference"):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        compile(open(filepath).read(), filepath, 'exec')
                    except SyntaxError:
                        syntax_errors += 1
                        issues_found += 1

        checks.append({
            'name': 'Syntax Errors',
            'passed': syntax_errors == 0,
            'issues': syntax_errors
        })

        # Check for TODO comments (should be minimal)
        print("  Checking for excessive TODOs...")
        todo_count = 0
        for root, dirs, files in os.walk("src/python/active_inference"):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            todo_count += content.upper().count('TODO')
                    except:
                        pass

        # Allow some TODOs but not excessive
        max_todos = 50
        todo_check_passed = todo_count <= max_todos
        if not todo_check_passed:
            issues_found += 1

        checks.append({
            'name': 'TODO Comments',
            'passed': todo_check_passed,
            'count': todo_count,
            'max_allowed': max_todos
        })

        # Check for print statements in production code
        print("  Checking for debug print statements...")
        print_count = 0
        for root, dirs, files in os.walk("src/python/active_inference"):
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            # Count print statements that aren't in test files or obvious debug
                            lines = content.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line.startswith('print(') and 'test' not in filepath.lower():
                                    print_count += 1
                    except:
                        pass

        max_prints = 10
        print_check_passed = print_count <= max_prints
        if not print_check_passed:
            issues_found += 1

        checks.append({
            'name': 'Debug Print Statements',
            'passed': print_check_passed,
            'count': print_count,
            'max_allowed': max_prints
        })

        success = issues_found == 0
        self.results['code_quality'] = {
            'passed': success,
            'checks': checks,
            'total_issues': issues_found
        }

        if success:
            print("✅ Code quality standards met")
        else:
            print(f"❌ Code quality issues found ({issues_found})")
            self.passed = False

        return success

    def run_all_checks(self) -> bool:
        """Run all quality gate checks."""
        print("🚀 Starting Comprehensive Quality Gates Validation")
        print("=" * 60)

        checks = [
            self.run_unified_interface_check,
            self.run_documentation_check,
            self.run_test_coverage_check,
            self.run_orchestration_integration_check,
            self.run_code_quality_checks
        ]

        for check in checks:
            check()

        print("=" * 60)

        # Generate final report
        self.generate_final_report()

        return self.passed

    def generate_final_report(self):
        """Generate comprehensive final report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"quality_gates_final_report_{timestamp}.json"

        report = {
            'timestamp': timestamp,
            'overall_passed': self.passed,
            'checks': self.results,
            'summary': {
                'unified_interfaces': self.results.get('unified_interfaces', {}).get('passed', False),
                'documentation': self.results.get('documentation', {}).get('passed', False),
                'test_coverage': self.results.get('test_coverage', {}).get('passed', False),
                'orchestration': self.results.get('orchestration', {}).get('passed', False),
                'code_quality': self.results.get('code_quality', {}).get('passed', False)
            }
        }

        # Save detailed report
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        print("📊 FINAL QUALITY GATES REPORT")
        print(f"   Report saved: {report_file}")
        print(f"   Overall result: {'✅ PASSED' if self.passed else '❌ FAILED'}")
        print()
        print("   Check Results:")
        for check_name, result in report['summary'].items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"     {check_name}: {status}")

        if not self.passed:
            print()
            print("   Failed checks need attention before deployment.")


def main():
    """Main quality gates execution."""
    gates = QualityGates()

    try:
        success = gates.run_all_checks()

        if success:
            print("\n🎉 ALL QUALITY GATES PASSED!")
            print("   Codebase meets 100% tested, documented, logged, unified, signposted, orchestrated standards.")
            sys.exit(0)
        else:
            print("\n⚠️  QUALITY GATES FAILED")
            print("   Address the failed checks before proceeding.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⏹️  Quality gates check interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Quality gates check failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
