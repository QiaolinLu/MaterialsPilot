"""
Base Agent class for Multi-Agent System
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AgentResponse:
    """Response from an agent"""
    success: bool
    content: str
    metadata: Dict[str, Any]
    reasoning: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system"""

    def __init__(self, llm_client, name: str):
        """
        Initialize base agent

        Args:
            llm_client: LLM client for generating responses
            name: Name of the agent
        """
        self.llm_client = llm_client
        self.name = name

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass

    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """
        Process input and generate response

        Args:
            input_data: Input data for the agent

        Returns:
            AgentResponse with the agent's output
        """
        pass

    def _generate(self, prompt: str, system_prompt: Optional[str] = None) -> tuple[str, int]:
        """
        Generate response using LLM

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override

        Returns:
            Tuple of (response, tokens_used)
        """
        if system_prompt is None:
            system_prompt = self.get_system_prompt()

        response, tokens = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt
        )

        return response, tokens
