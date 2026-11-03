"""
Generator Agent: Generates Python code based on execution plans
"""

from typing import Dict, Any, Optional
import json
from .base_agent import BaseAgent, AgentResponse
from .agent_config import AgentConfig


class GeneratorAgent(BaseAgent):
    """
    Generator Agent creates Python code to implement crystal structures

    Responsibilities:
    - Generate pymatgen code based on execution plan
    - Use RAG context if available
    - Follow best practices for structure generation
    - Ensure code is executable and creates 'structure' variable
    """

    def __init__(self, llm_client, rag_system=None, config: AgentConfig = None):
        super().__init__(llm_client, name="Generator")
        self.rag_system = rag_system
        self.config = config or AgentConfig.load()
        self._exemplar_context = ""

    def set_exemplar_context(self, context: str) -> None:
        """Set RAG-retrieved exemplar context for prompt injection."""
        self._exemplar_context = context

    def get_system_prompt(self) -> str:
        # Try loading from prompt template
        template = self.config.get_prompt_template('generator')
        if template:
            return self._build_prompt_from_template(template)

        # Fallback: built-in prompt
        return self._get_builtin_prompt()

    def _build_prompt_from_template(self, template: dict) -> str:
        """Build system prompt from externalized template."""
        validation_rules = self.config.format_validation_rules_for_prompt()
        principles = "\n".join(f"- {p}" for p in template.get('principles', []))
        exemplars = self._exemplar_context or ""

        return f"""{template.get('role', '')}

## Core Principles:
{principles}

## API Reference:
{template.get('api_reference', '')}

## Physical Validation:
{validation_rules}

{exemplars}

Output ONLY Python code, no markdown code blocks, no explanations."""

    def _get_builtin_prompt(self) -> str:
        """Fallback built-in prompt."""
        # Load dynamic validation rules from config
        validation_rules = self.config.format_validation_rules_for_prompt()

        return f"""You are a Generator Agent specialized in writing Python code for crystal structure generation.

Your role is to generate executable Python code using pymatgen based on execution plans.

## Your Responsibilities:
1. Generate clean, executable Python code
2. Follow the execution plan step-by-step
3. Use pymatgen best practices
4. Ensure physical validity (no atom overlap, correct composition)

## Code Requirements:
1. Start directly with imports (no markdown, no explanations)
2. Create a variable called 'structure' with the final Structure object
3. Use proper error handling for edge cases
4. Validate minimum interatomic distances
5. Verify atom counts and composition

## Pymatgen API Patterns (common examples, not exhaustive):

### Creating a structure from spacegroup (works for ANY crystal system):
```python
from pymatgen.core import Lattice, Structure

# General pattern: specify space group, lattice, species, and Wyckoff positions
structure = Structure.from_spacegroup(
    "Fm-3m",  # space group symbol or number
    Lattice.cubic(5.43),  # lattice matching the crystal system
    ["Na", "Cl"],  # species
    [[0, 0, 0], [0.5, 0.5, 0.5]]  # fractional coordinates (Wyckoff positions)
)
```

### Creating a supercell:
```python
structure.make_supercell([2, 2, 2])  # Creates 2x2x2 supercell
```

### Creating a vacancy:
```python
structure.remove_sites([0])  # Remove atom at index 0
```

### Substitution:
```python
structure[5] = "N"  # Replace atom at index 5
```

### Multi-component / alloy structures:
```python
# For disordered alloys or solid solutions, place atoms on sites directly
# For ordered alloys, use the appropriate space group and Wyckoff positions
```

These are common patterns. For novel structure types (Heusler, clathrate, Ruddlesden-Popper,
high-entropy alloys, etc.), use Structure.from_spacegroup() with the appropriate space group,
lattice parameters, and Wyckoff positions from crystallographic knowledge.

## Physical Validation:
{validation_rules}

## CRITICAL RULES:
- species list and coords list MUST have the same length
- Fractional coordinates should be in [0, 1) range
- Always verify the structure is created before returning

Output ONLY Python code, no markdown code blocks, no explanations."""

    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Generate code based on plan and optional RAG context

        Args:
            input_data: Dict with 'plan', 'task_description', optional 'rag_context', 'feedback'

        Returns:
            AgentResponse with generated code
        """
        plan = input_data.get('plan', {})
        task_description = input_data.get('task_description', '')
        task_input = input_data.get('task_input', {})
        rag_context = input_data.get('rag_context', '')
        feedback = input_data.get('feedback', '')  # From Critic/Refiner

        # Build prompt
        prompt_parts = []

        if task_description:
            prompt_parts.append(f"Task: {task_description}")

        if plan:
            prompt_parts.append(f"\nExecution Plan:\n{json.dumps(plan, indent=2)}")

        if task_input:
            prompt_parts.append(f"\nStructured Task Input:\n{json.dumps(task_input, indent=2)}")

        if rag_context:
            prompt_parts.append(f"\nRetrieved Knowledge:\n{rag_context}")

        if feedback:
            prompt_parts.append(f"\nFeedback from previous attempt:\n{feedback}")

        prompt_parts.append("\nGenerate Python code to implement this task:")

        prompt = "\n".join(prompt_parts)

        # Generate code
        response, tokens = self._generate(prompt)

        # Clean up response (remove markdown if present)
        code = self._extract_code(response)

        return AgentResponse(
            success=True,
            content=code,
            metadata={
                'tokens_used': tokens,
                'has_feedback': bool(feedback)
            },
            reasoning=f"Generated code with {len(code.split(chr(10)))} lines"
        )

    def _extract_code(self, response: str) -> str:
        """Extract Python code from response, removing markdown if present"""
        response = response.strip()

        if '```python' in response:
            code_part = response.split('```python', 1)[1].split('```', 1)[0]
            return code_part.strip()

        if '```' in response:
            parts = response.split('```')
            if len(parts) >= 3:
                return parts[1].strip()

        lines = response.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('from ') or stripped.startswith('import '):
                return '\n'.join(lines[index:]).strip()

        return response.strip()
