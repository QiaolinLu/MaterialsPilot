"""
Structure Evaluation Module

Provides unified evaluation framework for crystal structure generation tasks.
"""

from .base import BaseEvaluator, EvaluationResult
from .general import GeneralEvaluator
from .grain_boundary import GrainBoundaryEvaluator
from .interface import InterfaceEvaluator
from .defect import DefectEvaluator
from .adsorption import AdsorptionEvaluator
from .mlip import MLIPEvaluator
from .protocol import evaluate_result_protocol, evaluate_structure_protocol, summarize_protocol_results

__all__ = [
    'BaseEvaluator',
    'EvaluationResult',
    'GeneralEvaluator',
    'GrainBoundaryEvaluator',
    'InterfaceEvaluator',
    'DefectEvaluator',
    'AdsorptionEvaluator',
    'MLIPEvaluator',
    'evaluate_result_protocol',
    'evaluate_structure_protocol',
    'summarize_protocol_results',
]
