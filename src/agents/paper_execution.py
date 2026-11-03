"""Execution and result types shared by the single paper workflow.

Execution semantics are retained from the original subprocess executor: partial
structures are reported as such and the paper loop excludes them on an error.
"""
import io
import json
import logging
import textwrap
import tokenize
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationResult:
    """Result from multi-agent orchestration"""
    success: bool
    final_code: str
    structure: Optional[Any]
    iterations: int
    total_tokens: int
    plan: Dict[str, Any]
    evaluations: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class PaperExecution:
    """Subprocess execution and serialization; no policy or repair heuristics."""

    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute generated code safely in a subprocess sandbox.

        1. Syntax pre-check via compile() — fast fail on SyntaxError
        2. Append JSON serialization so the structure is written to disk
        3. Execute in SandboxExecutor (subprocess + 60s timeout)
        4. Load the JSON back into a pymatgen Structure
        """
        from pymatgen.core import Structure
        from src.core.sandbox_executor import SandboxExecutor

        # 1. Syntax pre-check
        try:
            compile(code, '<generated>', 'exec')
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return {
                'success': False,
                'output': f"SyntaxError: {e}",
                'structure': None,
            }

        non_ascii_identifier = self._find_non_ascii_identifier(code)
        if non_ascii_identifier:
            message = (
                "Generated code contains a non-ASCII Python identifier "
                f"{non_ascii_identifier!r}. Regenerate ASCII-only executable "
                "Python code and replace any prose placeholders with numeric "
                "or string literals."
            )
            logger.error(message)
            return {
                'success': False,
                'output': message,
                'structure': None,
            }

        # 2. Wrap code to serialize structure to JSON. If generated code creates
        # `structure` and then fails a final self-check assertion, preserve the
        # partial structure so deterministic/MLIP repair can inspect it. JSON is
        # used here instead of CIF because CIF round-tripping can collapse
        # periodic duplicate sites into occupancies > 1 before the critic has a
        # chance to diagnose or repair the generated Structure.
        output_filename = '_output_structure.json'
        wrapped_code = (
            "import traceback as _materialspilot_traceback\n"
            "import json as _materialspilot_json\n"
            "_materialspilot_execution_error = None\n"
            "try:\n"
            + textwrap.indent(code, "    ")
            + "\n"
            "except Exception:\n"
            "    _materialspilot_execution_error = _materialspilot_traceback.format_exc()\n"
            "finally:\n"
            "    if 'structure' in globals() and structure is not None:\n"
            "        with open('" + output_filename + "', 'w') as _materialspilot_f:\n"
            "            _materialspilot_json.dump(structure.as_dict(), _materialspilot_f)\n"
            "    if _materialspilot_execution_error:\n"
            "        print('__MATERIALSPILOT_PARTIAL_ERROR_START__')\n"
            "        print(_materialspilot_execution_error)\n"
            "        print('__MATERIALSPILOT_PARTIAL_ERROR_END__')\n"
        )

        # 3. Execute in sandbox
        sandbox = SandboxExecutor(timeout=60)
        success, file_content, result_info = sandbox.execute_with_file_output(
            wrapped_code, output_filename
        )

        if not success:
            partial_error = self._extract_partial_execution_error(
                result_info.get('stdout', '')
            )
            error_msg = partial_error or result_info.get('error', 'Unknown execution error')
            logger.error(f"Sandbox execution failed: {error_msg}")
            return {
                'success': False,
                'output': error_msg,
                'structure': None,
                'partial_execution_error': bool(partial_error),
            }

        # 4. Load structure from JSON content
        try:
            structure = Structure.from_dict(json.loads(file_content))

            output = result_info.get('stdout', '')
            partial_error = self._extract_partial_execution_error(output)
            if partial_error:
                output = partial_error
            return {
                'success': True,
                'output': output,
                'structure': structure,
                'partial_execution_error': bool(partial_error),
            }

        except Exception as e:
            logger.error(f"Failed to load structure from JSON: {e}")
            return {
                'success': False,
                'output': f"Structure created but failed to load JSON: {e}",
                'structure': None,
            }

    @staticmethod
    def _extract_partial_execution_error(output: str) -> str:
        start = "__MATERIALSPILOT_PARTIAL_ERROR_START__"
        end = "__MATERIALSPILOT_PARTIAL_ERROR_END__"
        if start not in output or end not in output:
            return ""
        try:
            return output.split(start, 1)[1].split(end, 1)[0].strip()
        except Exception:
            return ""

    @staticmethod
    def _find_non_ascii_identifier(code: str) -> str:
        try:
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            for token in tokens:
                if token.type == tokenize.NAME and not token.string.isascii():
                    return token.string
        except tokenize.TokenError:
            return ""
        return ""

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert nested metadata to JSON-serializable values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {
                str(PaperExecution._json_safe(k)): PaperExecution._json_safe(v)
                for k, v in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [PaperExecution._json_safe(v) for v in value]

        try:
            import numpy as np

            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
        except Exception:
            pass

        if hasattr(value, "to_dict"):
            try:
                return PaperExecution._json_safe(value.to_dict())
            except Exception:
                pass

        return str(value)
