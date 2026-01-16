"""Language Module for natural language command parsing and understanding.

This module provides natural language processing capabilities using GPT-3.5 for robotic task automation.
It enables the agent to:
- Parse natural language commands into structured representations
- Extract high-level intent from user commands
- Check for ambiguity and generate clarification questions
- Generate human-readable confirmations

The module uses LangChain with OpenAI's GPT models and Pydantic for structured output parsing.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParsedCommand(BaseModel):
    """Structured representation of a parsed robotic command."""
    
    action: str = Field(description="The primary action to perform (e.g., 'pick', 'place', 'move', 'sort')")
    target_object: Optional[str] = Field(default=None, description="The object to manipulate")
    destination: Optional[str] = Field(default=None, description="Where to move or place the object")
    constraints: List[str] = Field(default_factory=list, description="Additional constraints or modifiers")
    confidence: float = Field(default=1.0, description="Confidence in the parsing (0-1)")


class AmbiguityCheck(BaseModel):
    """Result of ambiguity checking for a command."""
    
    is_ambiguous: bool = Field(description="Whether the command is ambiguous")
    ambiguity_score: float = Field(description="Degree of ambiguity (0=clear, 1=very ambiguous)")
    unclear_aspects: List[str] = Field(default_factory=list, description="Specific unclear elements")
    clarification_questions: List[str] = Field(default_factory=list, description="Questions to resolve ambiguity")


class LanguageModule:
    """
    Language understanding module for parsing natural language commands.
    
    Uses GPT-3.5 to extract structured information from natural language
    and assess command clarity.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.3
    ):
        """
        Initialize the LanguageModule with OpenAI LLM.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model_name: OpenAI model to use
            temperature: Sampling temperature (lower = more deterministic)
        
        Raises:
            ValueError: If API key is not provided or found in environment
        """
        try:
            # Get API key from parameter or environment variable
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set in environment")
            
            # Initialize OpenAI LLM with low temperature for deterministic parsing
            # Lower temperature (0.3) ensures consistent structured output
            self.llm = ChatOpenAI(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature  # Low temperature for more deterministic parsing
            )
            
            # Setup parsers
            # Pydantic parsers ensure LLM outputs match expected schema
            self.command_parser = PydanticOutputParser(pydantic_object=ParsedCommand)
            self.ambiguity_parser = PydanticOutputParser(pydantic_object=AmbiguityCheck)
            
            logger.info(f"LanguageModule initialized with {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LanguageModule: {e}")
            raise
    
    def parse_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> ParsedCommand:
        """
        Parse a natural language command into structured format.
        
        Args:
            command: Natural language command string
            context: Optional context (e.g., visual information, previous commands)
        
        Returns:
            ParsedCommand object with extracted action, target, destination, etc.
        
        Raises:
            ValueError: If command is empty or invalid
            RuntimeError: If parsing fails
        """
        try:
            if not command or not command.strip():
                raise ValueError("Command cannot be empty")
            
            # Build context string
            # Incorporate visual information to improve parsing accuracy
            context_str = ""
            if context:
                # Extract detected objects from visual context
                if "detected_objects" in context:
                    objects = [obj["description"] for obj in context["detected_objects"][:5]]
                    context_str += f"\nVisible objects: {', '.join(objects)}"
                # Add workspace description for spatial understanding
                if "workspace_description" in context:
                    context_str += f"\nWorkspace: {context['workspace_description']}"
            
            # Create parsing prompt
            # Use structured prompting to guide GPT towards consistent output format
            template = """You are a robotic command parser. Extract structured information from natural language commands.

Command: {command}
{context}

Extract the following information:
- action: The primary action (pick, place, move, sort, grasp, release, etc.)
- target_object: The object to manipulate (include color/type if specified)
- destination: Where to move/place the object (if applicable)
- constraints: Any additional requirements (carefully, quickly, avoid obstacles, etc.)
- confidence: Your confidence in this parsing (0.0 to 1.0)

{format_instructions}

Provide your response as valid JSON matching the schema."""

            prompt = PromptTemplate(
                template=template,
                input_variables=["command", "context"],
                partial_variables={"format_instructions": self.command_parser.get_format_instructions()}
            )
            
            # Parse command
            formatted_prompt = prompt.format(command=command, context=context_str)
            response = self.llm.invoke(formatted_prompt)
            parsed = self.command_parser.parse(response.content)
            
            logger.info(f"Parsed command: action={parsed.action}, target={parsed.target_object}, "
                       f"destination={parsed.destination}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse command '{command}': {e}")
            # Return a fallback parsed command
            return ParsedCommand(
                action="unknown",
                target_object=None,
                destination=None,
                constraints=[],
                confidence=0.0
            )
    
    def check_ambiguity(
        self,
        command: str,
        parsed_command: Optional[ParsedCommand] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AmbiguityCheck:
        """
        Check if a command is ambiguous and needs clarification.
        
        Args:
            command: Original natural language command
            parsed_command: Optional pre-parsed command structure
            context: Optional context information
        
        Returns:
            AmbiguityCheck object with ambiguity assessment
        
        Raises:
            RuntimeError: If ambiguity checking fails
        """
        try:
            # Build context
            context_str = ""
            if parsed_command:
                context_str += f"\nParsed as: action={parsed_command.action}, "
                context_str += f"target={parsed_command.target_object}, dest={parsed_command.destination}"
            
            if context and "detected_objects" in context:
                objects = [obj["description"] for obj in context["detected_objects"][:5]]
                context_str += f"\nVisible objects: {', '.join(objects)}"
            
            # Create ambiguity checking prompt
            template = """You are analyzing robotic commands for ambiguity and clarity.

Command: "{command}"
{context}

Assess whether this command is clear enough for a robot to execute safely and correctly.

Consider:
- Is the target object clearly specified?
- Is the action unambiguous?
- If placement is involved, is the destination clear?
- Are there multiple possible interpretations?
- Is additional information needed?

{format_instructions}

Provide your assessment as valid JSON matching the schema."""

            prompt = PromptTemplate(
                template=template,
                input_variables=["command", "context"],
                partial_variables={"format_instructions": self.ambiguity_parser.get_format_instructions()}
            )
            
            # Check ambiguity
            formatted_prompt = prompt.format(command=command, context=context_str)
            response = self.llm.invoke(formatted_prompt)
            ambiguity_check = self.ambiguity_parser.parse(response.content)
            
            logger.info(f"Ambiguity check: is_ambiguous={ambiguity_check.is_ambiguous}, "
                       f"score={ambiguity_check.ambiguity_score:.2f}")
            
            return ambiguity_check
            
        except Exception as e:
            logger.error(f"Failed to check ambiguity: {e}")
            # Return a conservative fallback
            return AmbiguityCheck(
                is_ambiguous=True,
                ambiguity_score=0.8,
                unclear_aspects=["Unable to assess command clarity"],
                clarification_questions=["Could you please rephrase the command?"]
            )
    
    def extract_intent(self, command: str) -> Dict[str, Any]:
        """
        Extract high-level intent from a command.
        
        Args:
            command: Natural language command
        
        Returns:
            Dictionary with intent classification and confidence
        """
        try:
            template = """Classify the intent of this robotic command into one of these categories:
- MANIPULATION: Picking, placing, grasping objects
- NAVIGATION: Moving to locations
- INSPECTION: Looking at or analyzing objects
- SORTING: Organizing or categorizing objects
- QUERY: Asking for information
- OTHER: Other intents

Command: "{command}"

Respond with JSON: {{"intent": "CATEGORY", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

            prompt = template.format(command=command)
            response = self.llm.invoke(prompt)
            
            # Parse JSON response
            result = json.loads(response.content)
            logger.debug(f"Extracted intent: {result['intent']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract intent: {e}")
            return {
                "intent": "OTHER",
                "confidence": 0.0,
                "reasoning": "Intent extraction failed"
            }
    
    def generate_confirmation(self, parsed_command: ParsedCommand) -> str:
        """
        Generate a natural language confirmation of the parsed command.
        
        Args:
            parsed_command: Parsed command structure
        
        Returns:
            Human-readable confirmation string
        """
        try:
            parts = [f"I will {parsed_command.action}"]
            
            if parsed_command.target_object:
                parts.append(f"the {parsed_command.target_object}")
            
            if parsed_command.destination:
                parts.append(f"to {parsed_command.destination}")
            
            if parsed_command.constraints:
                parts.append(f"({', '.join(parsed_command.constraints)})")
            
            confirmation = " ".join(parts) + "."
            
            if parsed_command.confidence < 0.7:
                confirmation += f" (Confidence: {parsed_command.confidence:.0%})"
            
            return confirmation
            
        except Exception as e:
            logger.error(f"Failed to generate confirmation: {e}")
            return "Command received but confirmation generation failed."

