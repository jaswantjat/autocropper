#!/usr/bin/env python3
"""
Test runner script for the Card Rectification API test suite.

This script can be used to run tests both locally and in deployment environments.
It handles dependency installation and provides detailed test reporting.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def install_test_dependencies():
    """Install test dependencies if not already available."""
    try:
        import pytest
        import numpy
        import cv2
        print("✓ Test dependencies already available")
        return True
    except ImportError as e:
        print(f"Installing missing test dependency: {e.name}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'pytest', 'numpy', 'opencv-python'
            ])
            print("✓ Test dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install test dependencies")
            return False


def run_tests(verbose=False, coverage=False, specific_test=None):
    """Run the test suite with specified options."""
    cmd = [sys.executable, '-m', 'pytest']
    
    if verbose:
        cmd.append('-v')
    else:
        cmd.append('-q')
    
    if coverage:
        try:
            import coverage
            cmd.extend(['--cov=app', '--cov=card_rectification', '--cov-report=html'])
        except ImportError:
            print("Coverage requested but pytest-cov not available. Installing...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytest-cov'])
            cmd.extend(['--cov=app', '--cov=card_rectification', '--cov-report=html'])
    
    if specific_test:
        cmd.append(specific_test)
    else:
        cmd.append('tests/')
    
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser(description='Run Card Rectification API tests')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Verbose test output')
    parser.add_argument('-c', '--coverage', action='store_true',
                       help='Generate coverage report')
    parser.add_argument('-t', '--test', type=str,
                       help='Run specific test file or test function')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install test dependencies before running')
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("Card Rectification API Test Suite")
    print("=" * 50)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_test_dependencies():
            sys.exit(1)
    
    # Run tests
    exit_code = run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        specific_test=args.test
    )
    
    if exit_code == 0:
        print("\n✓ All tests passed!")
        if args.coverage:
            print("Coverage report generated in htmlcov/")
    else:
        print(f"\n✗ Tests failed with exit code {exit_code}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
