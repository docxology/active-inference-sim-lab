# Scripts & Automation - AGENTS

## Module Overview

The `scripts` directory contains automation scripts and utilities for development, deployment, testing, and maintenance of the Active Inference Simulation Lab.

## Script Categories

### Development Scripts

**Environment Setup:**
```bash
#!/bin/bash
# scripts/setup_dev_environment.sh

set -e

echo "Setting up Active Inference development environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install package in development mode
pip install -e .

# Setup pre-commit hooks
pre-commit install

# Create local configuration
cp .env.example .env

echo "Development environment setup complete!"
echo "Run 'source venv/bin/activate' to activate the environment."
```

**Code Quality Tools:**
```python
#!/usr/bin/env python3
# scripts/run_quality_checks.py

"""
Run comprehensive code quality checks.
"""

import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a command and report results."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def main():
    """Run all quality checks."""

    checks = [
        ("black --check --diff src/ tests/", "Code formatting (Black)"),
        ("flake8 src/ tests/", "Linting (flake8)"),
        ("mypy src/", "Type checking (mypy)"),
        ("pytest tests/ -v --cov=src/ --cov-report=term-missing", "Unit tests"),
        ("bandit -r src/ -f json", "Security scanning (Bandit)"),
        ("safety check", "Dependency vulnerability check"),
    ]

    results = []
    for command, description in checks:
        success = run_command(command, description)
        results.append((description, success))

    # Summary
    print("\n" + "="*50)
    print("QUALITY CHECK SUMMARY")
    print("="*50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {description}")

    print(f"\nOverall: {passed}/{total} checks passed")

    if passed < total:
        print("\n❌ Some quality checks failed. Please fix the issues before committing.")
        sys.exit(1)
    else:
        print("\n✅ All quality checks passed!")

if __name__ == "__main__":
    main()
```

### Deployment Scripts

**Docker Build and Deploy:**
```bash
#!/bin/bash
# scripts/deploy_docker.sh

set -e

IMAGE_NAME="active-inference"
TAG=${1:-"latest"}
ENVIRONMENT=${2:-"development"}

echo "Building Active Inference Docker image..."
echo "Tag: $TAG"
echo "Environment: $ENVIRONMENT"

# Build multi-platform image
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    --tag $IMAGE_NAME:$TAG \
    --tag $IMAGE_NAME:$ENVIRONMENT \
    --push \
    .

echo "Image built and pushed: $IMAGE_NAME:$TAG"

# Deploy based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "Deploying to production..."
    kubectl set image deployment/active-inference active-inference=$IMAGE_NAME:$TAG
    kubectl rollout status deployment/active-inference
elif [ "$ENVIRONMENT" = "staging" ]; then
    echo "Deploying to staging..."
    kubectl set image deployment/active-inference-staging active-inference=$IMAGE_NAME:$TAG
    kubectl rollout status deployment/active-inference-staging
fi

echo "Deployment complete!"
```

**Kubernetes Deployment:**
```python
#!/usr/bin/env python3
# scripts/deploy_kubernetes.py

"""
Kubernetes deployment automation script.
"""

import argparse
import subprocess
import yaml
from pathlib import Path

def load_k8s_manifests(manifest_dir: str, version: str) -> list:
    """Load and template Kubernetes manifests."""

    manifests = []
    manifest_path = Path(manifest_dir)

    for yaml_file in manifest_path.glob("*.yaml"):
        with open(yaml_file, 'r') as f:
            manifest = yaml.safe_load(f)

        # Template version
        if isinstance(manifest, dict) and 'spec' in manifest:
            if 'template' in manifest['spec']:
                containers = manifest['spec']['template']['spec'].get('containers', [])
                for container in containers:
                    if 'image' in container:
                        # Replace image tag
                        image_parts = container['image'].split(':')
                        if len(image_parts) == 2:
                            container['image'] = f"{image_parts[0]}:{version}"

        manifests.append(manifest)

    return manifests

def deploy_to_kubernetes(manifests: list, namespace: str, dry_run: bool = False):
    """Deploy manifests to Kubernetes."""

    for manifest in manifests:
        # Convert back to YAML
        yaml_content = yaml.dump(manifest, default_flow_style=False)

        # Apply manifest
        cmd = f"kubectl apply -f - --namespace {namespace}"
        if dry_run:
            cmd += " --dry-run=client"
            print(f"Dry run: {cmd}")
            print(yaml_content)
            print("-" * 50)
        else:
            result = subprocess.run(cmd, input=yaml_content,
                                  shell=True, text=True, capture_output=True)

            if result.returncode != 0:
                print(f"Failed to apply manifest: {result.stderr}")
                return False

    return True

def wait_for_rollout(deployment_name: str, namespace: str, timeout: int = 300):
    """Wait for deployment rollout to complete."""

    cmd = f"kubectl rollout status deployment/{deployment_name} --namespace {namespace} --timeout {timeout}s"
    result = subprocess.run(cmd, shell=True)

    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Deploy Active Inference to Kubernetes")
    parser.add_argument("--version", required=True, help="Version to deploy")
    parser.add_argument("--namespace", default="active-inference", help="Kubernetes namespace")
    parser.add_argument("--manifest-dir", default="k8s", help="Directory containing manifests")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run only")
    parser.add_argument("--wait", action="store_true", help="Wait for rollout completion")

    args = parser.parse_args()

    print(f"Deploying version {args.version} to namespace {args.namespace}")

    # Load manifests
    manifests = load_k8s_manifests(args.manifest_dir, args.version)
    print(f"Loaded {len(manifests)} manifests")

    # Deploy
    success = deploy_to_kubernetes(manifests, args.namespace, args.dry_run)

    if not success:
        print("Deployment failed!")
        exit(1)

    if args.wait and not args.dry_run:
        print("Waiting for rollout completion...")
        rollout_success = wait_for_rollout("active-inference", args.namespace)

        if rollout_success:
            print("Deployment successful!")
        else:
            print("Deployment failed - rollout timeout")
            exit(1)
    else:
        print("Deployment initiated successfully!")

if __name__ == "__main__":
    main()
```

### Testing Scripts

**Automated Testing Pipeline:**
```python
#!/usr/bin/env python3
# scripts/run_tests.py

"""
Comprehensive test execution script.
"""

import argparse
import subprocess
import sys
from datetime import datetime
import json

def run_test_suite(test_type: str, verbose: bool = False, coverage: bool = True) -> dict:
    """Run a specific test suite."""

    base_cmd = ["python", "-m", "pytest"]

    if test_type == "unit":
        cmd = base_cmd + ["tests/unit/", "-v" if verbose else ""]
    elif test_type == "integration":
        cmd = base_cmd + ["tests/integration/", "-v" if verbose else ""]
    elif test_type == "performance":
        cmd = base_cmd + ["tests/performance/", "-v" if verbose else ""]
    elif test_type == "all":
        cmd = base_cmd + ["tests/", "-v" if verbose else ""]
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    # Add coverage reporting
    if coverage:
        cmd.extend(["--cov=src/", "--cov-report=html", "--cov-report=term-missing"])

    # Run tests
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = datetime.now()

    test_results = {
        'test_type': test_type,
        'return_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'duration': (end_time - start_time).total_seconds(),
        'timestamp': start_time.isoformat()
    }

    return test_results

def run_quality_checks() -> dict:
    """Run code quality checks."""

    checks = {
        'linting': ['flake8', 'src/', 'tests/'],
        'formatting': ['black', '--check', '--diff', 'src/', 'tests/'],
        'type_checking': ['mypy', 'src/'],
        'security': ['bandit', '-r', 'src/', '-f', 'json']
    }

    results = {}

    for check_name, cmd in checks.items():
        print(f"Running {check_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        results[check_name] = {
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    return results

def generate_test_report(test_results: dict, quality_results: dict) -> str:
    """Generate comprehensive test report."""

    report = {
        'timestamp': datetime.now().isoformat(),
        'test_results': test_results,
        'quality_results': quality_results,
        'summary': {
            'tests_passed': all(r['return_code'] == 0 for r in test_results.values()),
            'quality_passed': all(r['return_code'] == 0 for r in quality_results.values()),
            'overall_success': False
        }
    }

    report['summary']['overall_success'] = (
        report['summary']['tests_passed'] and report['summary']['quality_passed']
    )

    return json.dumps(report, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Run Active Inference tests")
    parser.add_argument("test_type", choices=["unit", "integration", "performance", "all"],
                       help="Type of tests to run")
    parser.add_argument("--no-coverage", action="store_true",
                       help="Skip coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--quality-checks", action="store_true",
                       help="Run code quality checks")
    parser.add_argument("--report-file", help="Save report to JSON file")

    args = parser.parse_args()

    test_results = {}
    quality_results = {}

    # Run tests
    if args.test_type in ["unit", "integration", "performance", "all"]:
        test_results[args.test_type] = run_test_suite(
            args.test_type, args.verbose, not args.no_coverage
        )

    # Run quality checks
    if args.quality_checks:
        quality_results = run_quality_checks()

    # Generate report
    if args.report_file:
        report = generate_test_report(test_results, quality_results)
        with open(args.report_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {args.report_file}")

    # Exit with appropriate code
    test_success = all(r['return_code'] == 0 for r in test_results.values())
    quality_success = all(r['return_code'] == 0 for r in quality_results.values())

    if test_success and (not args.quality_checks or quality_success):
        print("✅ All checks passed!")
        sys.exit(0)
    else:
        print("❌ Some checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Monitoring and Maintenance Scripts

**System Health Check:**
```python
#!/usr/bin/env python3
# scripts/health_check.py

"""
System health monitoring and automated maintenance script.
"""

import psutil
import requests
import time
from datetime import datetime, timedelta
import subprocess
import json

class SystemHealthChecker:
    """Comprehensive system health checker."""

    def __init__(self, services=None):
        self.services = services or {
            'active_inference_api': 'http://localhost:8000/health',
            'prometheus': 'http://localhost:9090/-/healthy',
            'grafana': 'http://localhost:3000/api/health'
        }

        self.health_history = []
        self.alerts = []

    def check_system_health(self) -> dict:
        """Perform comprehensive system health check."""

        health_report = {
            'timestamp': datetime.now().isoformat(),
            'system': self._check_system_resources(),
            'services': {},
            'processes': self._check_processes(),
            'network': self._check_network(),
            'storage': self._check_storage(),
            'overall_status': 'healthy'
        }

        # Check services
        for service_name, health_url in self.services.items():
            health_report['services'][service_name] = self._check_service_health(
                service_name, health_url
            )

        # Determine overall status
        if any(s.get('status') != 'healthy' for s in health_report['services'].values()):
            health_report['overall_status'] = 'degraded'
        elif health_report['system']['status'] != 'healthy':
            health_report['overall_status'] = 'degraded'

        # Store in history
        self.health_history.append(health_report)

        # Keep only last 100 reports
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]

        return health_report

    def _check_system_resources(self) -> dict:
        """Check system resource usage."""

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        status = 'healthy'
        issues = []

        if cpu_percent > 90:
            status = 'critical'
            issues.append('.1f')
        elif cpu_percent > 70:
            status = 'warning'
            issues.append('.1f')

        if memory.percent > 90:
            status = 'critical'
            issues.append('.1f')
        elif memory.percent > 80:
            status = 'warning'
            issues.append('.1f')

        if disk.percent > 95:
            status = 'critical'
            issues.append('.1f')

        return {
            'status': status,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'issues': issues
        }

    def _check_service_health(self, service_name: str, health_url: str) -> dict:
        """Check individual service health."""

        try:
            response = requests.get(health_url, timeout=5)

            if response.status_code == 200:
                try:
                    health_data = response.json()
                    return {
                        'status': 'healthy',
                        'response_time': response.elapsed.total_seconds(),
                        'details': health_data
                    }
                except:
                    return {
                        'status': 'healthy',
                        'response_time': response.elapsed.total_seconds()
                    }
            else:
                return {
                    'status': 'unhealthy',
                    'http_status': response.status_code,
                    'response_time': response.elapsed.total_seconds()
                }

        except requests.exceptions.RequestException as e:
            return {
                'status': 'unreachable',
                'error': str(e)
            }

    def _check_processes(self) -> dict:
        """Check important system processes."""

        important_processes = ['python', 'prometheus', 'grafana-server']
        running_processes = []

        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if any(imp in proc.info['name'].lower() for imp in important_processes):
                    running_processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent']
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        return {
            'running_processes': running_processes,
            'expected_processes': important_processes
        }

    def _check_network(self) -> dict:
        """Check network connectivity and performance."""

        try:
            # Basic connectivity check
            response = requests.get('https://httpbin.org/status/200', timeout=5)

            return {
                'status': 'healthy' if response.status_code == 200 else 'degraded',
                'latency_ms': response.elapsed.total_seconds() * 1000,
                'external_connectivity': True
            }

        except requests.exceptions.RequestException:
            return {
                'status': 'degraded',
                'external_connectivity': False
            }

    def _check_storage(self) -> dict:
        """Check storage health."""

        disk = psutil.disk_usage('/')

        return {
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'usage_percent': disk.percent,
            'status': 'healthy' if disk.percent < 90 else 'warning' if disk.percent < 95 else 'critical'
        }

    def perform_maintenance(self) -> dict:
        """Perform automated maintenance tasks."""

        maintenance_report = {
            'timestamp': datetime.now().isoformat(),
            'tasks': []
        }

        # Clean up old log files
        try:
            result = subprocess.run(['find', '/var/log', '-name', '*.log',
                                   '-mtime', '+30', '-delete'],
                                  capture_output=True, text=True)
            maintenance_report['tasks'].append({
                'task': 'log_cleanup',
                'status': 'completed' if result.returncode == 0 else 'failed',
                'details': result.stdout if result.returncode == 0 else result.stderr
            })
        except Exception as e:
            maintenance_report['tasks'].append({
                'task': 'log_cleanup',
                'status': 'error',
                'details': str(e)
            })

        # Restart unhealthy services (example)
        health_report = self.check_system_health()
        for service_name, service_health in health_report['services'].items():
            if service_health['status'] != 'healthy':
                maintenance_report['tasks'].append({
                    'task': f'restart_{service_name}',
                    'status': 'pending',
                    'reason': f'Service {service_name} is unhealthy'
                })

        return maintenance_report

def main():
    """Main health check and maintenance function."""

    checker = SystemHealthChecker()

    # Perform health check
    health_report = checker.check_system_health()

    print("System Health Report:")
    print(json.dumps(health_report, indent=2))

    # Perform maintenance if issues found
    if health_report['overall_status'] != 'healthy':
        print("\\nPerforming maintenance...")
        maintenance_report = checker.perform_maintenance()
        print(json.dumps(maintenance_report, indent=2))

if __name__ == "__main__":
    main()
```

## Script Organization

### Directory Structure

```
scripts/
├── development/          # Development workflow scripts
│   ├── setup_dev.sh
│   ├── run_quality_checks.py
│   └── update_dependencies.py
├── deployment/           # Deployment automation scripts
│   ├── deploy_docker.sh
│   ├── deploy_kubernetes.py
│   └── rollback_deployment.sh
├── testing/              # Testing and validation scripts
│   ├── run_tests.py
│   ├── performance_test.py
│   └── integration_test.py
├── monitoring/           # Monitoring and maintenance scripts
│   ├── health_check.py
│   ├── log_analysis.py
│   └── backup_database.sh
├── utilities/            # General utility scripts
│   ├── generate_docs.py
│   ├── clean_build_artifacts.sh
│   └── version_bump.py
└── ci_cd/                # CI/CD pipeline scripts
    ├── github_actions/
    ├── jenkins/
    └── argo_cd/
```

### Script Standards

**Header Template:**
```bash
#!/bin/bash
# scripts/template.sh
#
# Description: Brief description of what this script does
# Author: Author name
# Date: YYYY-MM-DD
# Version: 1.0.0
#
# Usage: ./script.sh [options]
#
# Options:
#   -h, --help     Show help message
#   -v, --verbose  Enable verbose output
#   --dry-run      Show what would be done without doing it
```

**Error Handling:**
```bash
#!/bin/bash

set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit if any command in a pipeline fails

# Error handling function
error_exit() {
    echo "Error: $1" >&2
    exit 1
}

# Trap errors
trap 'error_exit "Script failed at line $LINENO"' ERR
```

## Script Testing

### Automated Script Testing

```python
#!/usr/bin/env python3
# scripts/test_scripts.py

"""
Test automation scripts for correctness and safety.
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

def test_script_execution(script_path, args=None, expect_failure=False):
    """Test script execution."""

    cmd = [script_path]
    if args:
        cmd.extend(args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if expect_failure:
            assert result.returncode != 0, f"Expected {script_path} to fail"
        else:
            assert result.returncode == 0, f"Script {script_path} failed: {result.stderr}"

        return True

    except subprocess.TimeoutExpired:
        assert False, f"Script {script_path} timed out"
    except FileNotFoundError:
        assert False, f"Script {script_path} not found"

def test_deployment_script():
    """Test deployment script with dry run."""

    deployment_script = "scripts/deploy_kubernetes.py"

    # Test help
    test_script_execution(deployment_script, ["--help"])

    # Test dry run (would need mock k8s environment)
    # test_script_execution(deployment_script, ["--version", "1.0.0", "--dry-run"])

def test_quality_checks():
    """Test quality check script."""

    quality_script = "scripts/run_quality_checks.py"

    # Should run without errors (may fail checks, but script should work)
    try:
        result = subprocess.run([sys.executable, quality_script],
                              capture_output=True, text=True, timeout=300)
        # Script itself should execute (checks may fail)
        assert result.returncode in [0, 1], f"Quality script failed to execute: {result.stderr}"
    except subprocess.TimeoutExpired:
        assert False, "Quality checks timed out"

def test_health_check():
    """Test health check script."""

    health_script = "scripts/health_check.py"

    result = subprocess.run([sys.executable, health_script],
                          capture_output=True, text=True, timeout=30)

    # Should complete successfully
    assert result.returncode == 0, f"Health check failed: {result.stderr}"

    # Should produce valid JSON output
    import json
    health_data = json.loads(result.stdout)
    assert 'timestamp' in health_data
    assert 'system' in health_data

def test_script_safety():
    """Test scripts don't perform dangerous operations without confirmation."""

    dangerous_scripts = [
        "scripts/deploy_kubernetes.py",
        "scripts/deploy_docker.sh"
    ]

    for script in dangerous_scripts:
        if Path(script).exists():
            # Scripts should require explicit confirmation or have dry-run modes
            with open(script, 'r') as f:
                content = f.read()
                # Check for safety measures
                assert '--dry-run' in content or '--confirm' in content, \
                    f"Script {script} lacks safety measures"

def main():
    """Run all script tests."""

    tests = [
        test_deployment_script,
        test_quality_checks,
        test_health_check,
        test_script_safety
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print(f"\\nResults: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Future Script Enhancements

### Advanced Automation
- **AI-Powered Operations**: ML-based deployment and scaling decisions
- **Self-Healing Systems**: Automated issue detection and resolution
- **Predictive Maintenance**: Forecasting system issues before they occur
- **Multi-Cloud Operations**: Cross-cloud deployment and management

### Integration Improvements
- **GitOps Integration**: Git-based deployment and configuration management
- **Infrastructure as Code**: Declarative infrastructure management
- **Service Mesh Integration**: Advanced service communication management
- **Event-Driven Automation**: Event-based triggers for automated actions

### Security Enhancements
- **Automated Security Patching**: Continuous vulnerability management
- **Compliance Automation**: Automated compliance checking and reporting
- **Zero-Trust Automation**: Secure-by-default automation practices
- **Audit Automation**: Comprehensive audit trail generation

