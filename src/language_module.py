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
    """Structured representation of a parsed robotic command.
    
    This Pydantic model ensures consistent parsing of natural language commands
    into a structured format that can be used for action planning and execution.
    
    Attributes:
        action: Primary action verb (e.g., "pick", "place", "move", "sort", "grasp")
        target_object: Object to manipulate, including descriptors (e.g., "red block", "small cube")
        destination: Target location for placement or movement (e.g., "corner", "left side", "bin")
        constraints: Additional requirements or modifiers (e.g., "carefully", "quickly", "avoid obstacles")
        confidence: Parser's confidence in the interpretation, range 0.0-1.0
    
    Example:
        >>> cmd = ParsedCommand(
        ...     action="pick",
        ...     target_object="red block",
        ...     destination="corner",
        ...     constraints=["carefully"],
        ...     confidence=0.95
        ... )
        >>> print(f"{cmd.action} the {cmd.target_object}")  # "pick the red block"
    """
    
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
    and assess command clarity for safe robot execution.
    
    The module provides:
    - Command parsing into structured ParsedCommand objects
    - Ambiguity detection with clarification questions
    - Intent extraction for command classification
    - Natural language confirmation generation
    
    Example:
        >>> lang = LanguageModule(api_key="sk-...")
        >>> # Parse a command
        >>> cmd = lang.parse_command("Pick up the red block carefully")
        >>> print(cmd.action)  # "pick"
        >>> print(cmd.target_object)  # "red block"
        >>> print(cmd.constraints)  # ["carefully"]
        >>> 
        >>> # Check for ambiguity
        >>> ambiguity = lang.check_ambiguity("Pick it up", cmd)
        >>> if ambiguity.is_ambiguous:
        ...     print(ambiguity.clarification_questions[0])  # "Which object should I pick up?"
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
            # Use LLM to extract structured information from natural language
            formatted_prompt = prompt.format(command=command, context=context_str)
            response = self.llm.invoke(formatted_prompt)
            parsed = self.command_parser.parse(response.content)
            
            logger.info(f"Parsed command: action={parsed.action}, target={parsed.target_object}, "
                       f"destination={parsed.destination}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse command '{command}': {e}")
            # Return a fallback parsed command
            # This ensures the system can continue even if parsing fails
            # The low confidence (0.0) signals downstream components to handle carefully
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
            # Build context string
            # Include all available information to help assess clarity
            context_str = ""
            if context:
                if "detected_objects" in context:
                    objects = [obj["description"] for obj in context["detected_objects"][:5]]
                    context_str += f"\nVisible objects: {', '.join(objects)}"
                if "workspace_description" in context:
                    context_str += f"\nWorkspace: {context['workspace_description']}"
            
            # Create ambiguity checking prompt
            # Ask LLM to assess whether the command is clear enough for safe execution
            template = """Analyze the following robotic command for ambiguity.

Command: "{command}"

Parsed Information:
- Action: {action}
- Target: {target}
- Destination: {destination}
- Constraints: {constraints}
{context}

Assess if the command is clear enough for safe robot execution. Consider:
1. Are there multiple possible interpretations?
2. Is the target object clearly specified?
3. Is the destination (if needed) clear?
4. Are there any safety concerns due to unclear instructions?

{format_instructions}
"""

            # Format prompt with all context
            # Provide parsed command details and environmental context to the LLM
            formatted_prompt = template.format(
                command=command,
                action=parsed_command.action if parsed_command else "Not parsed yet",
                target=parsed_command.target_object if parsed_command and parsed_command.target_object else "Not specified",
                destination=parsed_command.destination if parsed_command and parsed_command.destination else "Not specified",
                constraints=", ".join(parsed_command.constraints) if parsed_command and parsed_command.constraints else "None",
                context=context_str,
                format_instructions=self.ambiguity_parser.get_format_instructions()
            )
            
            # Check ambiguity using the LLM
            # The LLM will output a structured response based on the AmbiguityCheck schema
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
            # Create intent extraction prompt
            # Classify the command into predefined intent categories
            template = """Extract the primary intent from this robotic command.

Command: "{command}"

Classify the intent into one of these categories:
- manipulation: Picking, placing, grasping, releasing objects
- navigation: Moving the robot base or arm to a location
- inspection: Looking at, examining, or identifying objects
- sorting: Organizing or categorizing objects
- assembly: Combining or constructing objects
- other: Any other intent

Respond with just the intent category (lowercase, single word).
"""

            # Extract intent using LLM
            # Simple classification task for routing or analytics
            prompt = template.format(command=command)
            response = self.llm.invoke(prompt)
            intent = response.content.strip().lower()
            
            # Validate intent
            # Ensure the response is one of the expected categories
            valid_intents = ["manipulation", "navigation", "inspection", "sorting", "assembly", "other"]
            if intent not in valid_intents:
                logger.warning(f"Unexpected intent '{intent}', defaulting to 'other'")
                intent = "other"
            
            logger.info(f"Extracted intent: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"Failed to extract intent: {e}")
            # Return default intent on error
            return "other"
    
    def generate_confirmation(self, parsed_command: ParsedCommand) -> str:
        """
        Generate a natural language confirmation of the parsed command.
        
        Args:
            parsed_command: Parsed command structure
        """
        try:
            # Create confirmation prompt
            # Generate human-friendly confirmation for user verification
            template = """Generate a natural, conversational confirmation message for this robotic command.

Parsed Command:
- Action: {action}
- Target: {target}
- Destination: {destination}
- Constraints: {constraints}

Generate a brief, friendly confirmation like:
"I'll {action} the {target} and place it {destination}."

Keep it natural and concise (1-2 sentences max).
"""

            # Format prompt with parsed command details
            # Provide all context for natural language generation
            prompt = template.format(
                action=parsed_command.action,
                target=parsed_command.target_object or "the object",
                destination=parsed_command.destination or "in the specified location",
                constraints=", ".join(parsed_command.constraints) if parsed_command.constraints else "none"
            )
            
            # Generate confirmation using LLM
            # LLM creates natural, user-friendly confirmation message
            response = self.llm.invoke(prompt)
            confirmation = response.content.strip()
            
            logger.debug(f"Generated confirmation: {confirmation}")
            return confirmation
            
        except Exception as e:
            logger.error(f"Failed to generate confirmation: {e}")
            # Return a simple fallback confirmation
            return f"I'll {parsed_command.action} the {parsed_command.target_object or 'object'}."
