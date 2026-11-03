"""Subprocess code execution with a timeout and temporary working directory.

This is process separation, not an OS security boundary for untrusted code.
"""

import subprocess
import tempfile
import os
import sys
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """Execute generated code using the caller's Python environment."""

    def __init__(self, timeout: int = 30, work_dir: str = None):
        self.timeout = timeout
        self.work_dir = work_dir

    def execute(self, code: str, context: Dict = None) -> Tuple[bool, str, Dict]:
        """
        Execute code in isolated environment

        Args:
            code: Python code to execute
            context: Optional context variables

        Returns:
            (success, output, result_dict)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, 'script.py')

            # Write code to temporary file
            with open(script_path, 'w') as f:
                f.write(code)

            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )

                if result.returncode == 0:
                    return True, result.stdout, {'stderr': result.stderr}
                else:
                    return False, result.stdout, {'error': result.stderr}

            except subprocess.TimeoutExpired:
                return False, "", {'error': f'Execution timeout ({self.timeout}s)'}
            except Exception as e:
                return False, "", {'error': str(e)}

    def execute_with_file_output(self, code: str, output_file: str = 'output.cif') -> Tuple[bool, str, Dict]:
        """
        Execute code and retrieve output file

        Args:
            code: Python code to execute
            output_file: Expected output file name

        Returns:
            (success, file_content, result_dict)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, 'script.py')
            output_path = os.path.join(tmpdir, output_file)

            # Write code to temporary file
            with open(script_path, 'w') as f:
                f.write(code)

            try:
                # Execute with timeout
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )

                # A partially written output must not hide a later exception.
                if result.returncode != 0:
                    return False, "", {
                        'error': result.stderr or f'Process exited with code {result.returncode}',
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                    }

                # Warnings on stderr alone do not indicate execution failure.
                if os.path.exists(output_path):
                    # Read output file
                    with open(output_path, 'r') as f:
                        file_content = f.read()

                    # Success - file was created (ignore warnings in stderr)
                    return True, file_content, {
                        'stdout': result.stdout,
                        'stderr': result.stderr,  # Include warnings for reference
                    }

                # returncode is 0 but no output file
                return False, "", {
                    'error': f'Output file {output_file} not created. stdout: {result.stdout[:200]}',
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                }

            except subprocess.TimeoutExpired:
                return False, "", {'error': f'Execution timeout ({self.timeout}s)'}
            except Exception as e:
                return False, "", {'error': str(e)}
