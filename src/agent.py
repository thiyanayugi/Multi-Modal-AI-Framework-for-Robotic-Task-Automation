"""Main Robotic Agent that orchestrates all modules for task automation.

This module provides the RoboticAgent class, which serves as the central orchestrator
for multi-modal robotic task automation. It integrates three key components:
- Vision Module: CLIP-based visual understanding for scene analysis
- Language Module: GPT-powered natural language command parsing
- RAG Module: Knowledge retrieval for manipulation strategies

The agent processes natural language commands, extracts visual context from images,
retrieves relevant knowledge, and generates detailed action plans for robot execution.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union
from PIL import Image
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

from .vision_module import VisionModule
from .language_module import LanguageModule, ParsedCommand
from .rag_module import RAGModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskResult:
    """Result of task processing.
    
    Encapsulates the complete output of the RoboticAgent's task processing pipeline,
    including the generated action plan, parsed command structure, visual context,
    retrieved knowledge, and any errors or ambiguity warnings.
    
    Attributes:
        success: Whether the task was processed successfully
        action_plan: Generated step-by-step action plan for robot execution
        parsed_command: Structured representation of the command
        visual_context: Extracted visual information from workspace image
        retrieved_knowledge: Relevant knowledge entries from RAG system
        ambiguity_check: Results of command clarity analysis
        error: Error message if processing failed
    
    Example:
        >>> result = agent.process_task("Pick up the red block")
        >>> if result.success:
        ...     print(result.action_plan)
    """
    
    def __init__(
        self,
        success: bool,
        action_plan: str,
        parsed_command: ParsedCommand,
        visual_context: Optional[Dict] = None,
        retrieved_knowledge: Optional[List[Dict]] = None,
        ambiguity_check: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.action_plan = action_plan
        self.parsed_command = parsed_command
        self.visual_context = visual_context
        self.retrieved_knowledge = retrieved_knowledge
        self.ambiguity_check = ambiguity_check
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        # Serialize TaskResult to dictionary format for JSON export or logging
        # This enables easy integration with ROS2 messages and external systems
        return {
            "success": self.success,
            "action_plan": self.action_plan,
            # Convert ParsedCommand Pydantic model to dict
            "parsed_command": {
                "action": self.parsed_command.action,
                "target_object": self.parsed_command.target_object,
                "destination": self.parsed_command.destination,
                "constraints": self.parsed_command.constraints,
                "confidence": self.parsed_command.confidence
            },
            # Include all context and metadata for debugging
            "visual_context": self.visual_context,
            "retrieved_knowledge": self.retrieved_knowledge,
            "ambiguity_check": self.ambiguity_check,
            "error": self.error
        }
    
    def __str__(self) -> str:
        """String representation."""
        # Provide concise string representation for logging
        if self.success:
            return f"TaskResult(success=True, action={self.parsed_command.action})"
        else:
            return f"TaskResult(success=False, error={self.error})"


class RoboticAgent:
    """
    Main robotic agent that integrates vision, language, and knowledge retrieval.
    
    Processes natural language commands and generates executable action plans
    using multimodal understanding and retrieval-augmented generation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        knowledge_base_path: Optional[str] = None,
        enable_vision: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the RoboticAgent with all modules.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment variable OPENAI_API_KEY)
            knowledge_base_path: Path to knowledge base JSON file containing manipulation strategies
                                (if None, starts with empty knowledge base)
            enable_vision: Whether to initialize vision module (set to False for text-only tasks)
            device: Device for vision module ('cuda' for GPU, 'cpu', or None for auto-detection)
        
        Raises:
            RuntimeError: If initialization fails (e.g., invalid API key, missing dependencies)
            
        Example:
            >>> # Initialize with API key and knowledge base
            >>> agent = RoboticAgent(
            ...     api_key="sk-...",
            ...     knowledge_base_path="./knowledge_base/manipulation_strategies.json",
            ...     enable_vision=True,
            ...     device="cuda"
            ... )
            >>> status = agent.get_system_status()
            >>> print(status["vision_module"])  # "enabled"
        """
        try:
            logger.info("Initializing RoboticAgent...")
            
            # Initialize language module first as it's required for all operations
            # This module handles natural language parsing and intent extraction
            self.language_module = LanguageModule(api_key=api_key)
            
            # Initialize RAG module for knowledge retrieval
            # Uses ChromaDB for semantic search over manipulation strategies
            self.rag_module = RAGModule()
            
            # Load knowledge base if provided
            # This populates the RAG system with domain-specific knowledge
            if knowledge_base_path and os.path.exists(knowledge_base_path):
                self.rag_module.load_knowledge_from_file(knowledge_base_path)
                logger.info(f"Loaded knowledge base from {knowledge_base_path}")
            
            # Initialize vision module if enabled
            # Vision is optional and can be disabled for text-only tasks
            self.vision_module = None
            if enable_vision:
                try:
                    self.vision_module = VisionModule(device=device)
                    logger.info("Vision module enabled")
                except Exception as e:
                    logger.warning(f"Failed to initialize vision module: {e}. Continuing without vision.")
            
            # Initialize LLM for action planning
            # This is separate from the language module's LLM to allow different temperatures
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.planner_llm = ChatOpenAI(
                api_key=api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.5  # Higher temperature for more creative action planning
            )
            
            logger.info("RoboticAgent initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RoboticAgent: {e}")
            raise RuntimeError(f"RoboticAgent initialization failed: {e}")
    
    def process_task(
        self,
        command: str,
        image: Optional[Union[str, Image.Image, np.ndarray]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """
        Process a robotic task from natural language command.
        
        This is the main entry point that orchestrates all modules:
        1. Parse command with LanguageModule
        2. Get visual context from VisionModule (if image provided)
        3. Retrieve relevant knowledge with RAGModule
        4. Check for ambiguity
        5. Generate action plan with LLM
        
        Args:
            command: Natural language command (e.g., "Pick up the red block")
            image: Optional image of workspace (file path, PIL Image, or numpy array)
            context: Optional additional context (e.g., {"robot_state": "idle"})
        
        Returns:
            TaskResult object with action plan and metadata
            
        Example:
            >>> agent = RoboticAgent(api_key="sk-...")
            >>> result = agent.process_task("Pick up the red block", image="workspace.jpg")
            >>> if result.success:
            ...     print(result.action_plan)
            ...     # Outputs: "1. Move to red block position\\n2. Open gripper\\n..."
        """
        try:
            logger.info(f"Processing task: '{command}'")
            
            # Step 1: Get visual context if image provided
            # Visual context helps the agent understand the workspace layout and object positions
            visual_context = None
            if image is not None and self.vision_module is not None:
                try:
                    visual_context = self.vision_module.get_visual_context(image)
                    logger.info(f"Visual context: {visual_context['workspace_description']}")
                except Exception as e:
                    # Continue without visual context if extraction fails
                    logger.warning(f"Failed to extract visual context: {e}")
            
            # Merge contexts
            # Combine visual context with any additional context provided by the caller
            full_context = context or {}
            if visual_context:
                full_context.update(visual_context)
            
            # Step 2: Parse command
            parsed_command = self.language_module.parse_command(command, full_context)
            logger.info(f"Parsed command: action={parsed_command.action}, "
                       f"target={parsed_command.target_object}")
            
            # Step 3: Retrieve relevant knowledge
            # RAG system provides domain-specific manipulation strategies
            retrieved_knowledge = []
            if parsed_command.action != "unknown":
                # Build retrieval query
                # Combine action, target object, and constraints for semantic search
                query_parts = [parsed_command.action]
                if parsed_command.target_object:
                    query_parts.append(parsed_command.target_object)
                if parsed_command.constraints:
                    # Limit to first 2 constraints to keep query focused
                    query_parts.extend(parsed_command.constraints[:2])
                
                retrieval_query = " ".join(query_parts)
                retrieved_knowledge = self.rag_module.retrieve(retrieval_query, top_k=3)
                logger.info(f"Retrieved {len(retrieved_knowledge)} knowledge entries")
            
            # Step 4: Check ambiguity
            # Assess whether the command is clear enough for safe execution
            ambiguity_check = self.language_module.check_ambiguity(
                command,
                parsed_command,
                full_context
            )
            
            # Reject highly ambiguous commands to prevent unsafe or incorrect actions
            # Threshold of 0.7 balances between safety and usability
            if ambiguity_check.is_ambiguous and ambiguity_check.ambiguity_score > 0.7:
                logger.warning(f"Command is ambiguous: {ambiguity_check.unclear_aspects}")
                return TaskResult(
                    success=False,
                    action_plan="",
                    parsed_command=parsed_command,
                    visual_context=visual_context,
                    retrieved_knowledge=retrieved_knowledge,
                    ambiguity_check=ambiguity_check.dict(),
                    error=f"Command is too ambiguous. Questions: {', '.join(ambiguity_check.clarification_questions)}"
                )
            
            # Step 5: Generate action plan
            action_plan = self._generate_action_plan(
                command,
                parsed_command,
                visual_context,
                retrieved_knowledge,
                full_context
            )
            
            logger.info("Task processing completed successfully")
            
            return TaskResult(
                success=True,
                action_plan=action_plan,
                parsed_command=parsed_command,
                visual_context=visual_context,
                retrieved_knowledge=retrieved_knowledge,
                ambiguity_check=ambiguity_check.dict()
            )
            
        except Exception as e:
            logger.error(f"Failed to process task: {e}")
            return TaskResult(
                success=False,
                action_plan="",
                parsed_command=ParsedCommand(action="error", confidence=0.0),
                error=str(e)
            )
    
    def _generate_action_plan(
        self,
        command: str,
        parsed_command: ParsedCommand,
        visual_context: Optional[Dict],
        retrieved_knowledge: List[Dict],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate detailed action plan using LLM with all available context.
        
        Args:
            command: Original command
            parsed_command: Parsed command structure
            visual_context: Visual information from scene
            retrieved_knowledge: Retrieved knowledge entries
            context: Additional context
        
        Returns:
            Detailed action plan as string
        """
        try:
            # Build context sections
            # Integrate visual information into the planning prompt
            visual_info = ""
            if visual_context:
                visual_info = f"""
Visual Context:
- Workspace: {visual_context.get('workspace_description', 'N/A')}
- Detected Objects: {', '.join([obj['description'] for obj in visual_context.get('detected_objects', [])[:5]])}
"""
            
            # Format retrieved knowledge for prompt
            # Provide relevant manipulation strategies to guide action planning
            knowledge_info = ""
            if retrieved_knowledge:
                knowledge_info = "Relevant Knowledge:\n"
                for i, entry in enumerate(retrieved_knowledge, 1):
                    # Truncate long knowledge entries to keep prompt concise
                    knowledge_info += f"{i}. {entry['content'][:200]}...\n"
            
            # Create planning prompt
            # Structured template ensures consistent and comprehensive action plans
            template = """You are a robotic task planner. Generate a detailed, step-by-step action plan for executing the given command.

Original Command: "{command}"

Parsed Command:
- Action: {action}
- Target Object: {target}
- Destination: {destination}
- Constraints: {constraints}
- Confidence: {confidence:.2f}

{visual_info}

{knowledge_info}

Generate a detailed action plan with:
1. Pre-execution checks (safety, workspace state)
2. Step-by-step execution sequence
3. Expected outcomes and verification
4. Error handling considerations

Be specific about movements, grip forces, speeds, and safety measures.
Format as a numbered list of concrete steps."""

            # Fill in template with all available context
            prompt = template.format(
                command=command,
                action=parsed_command.action,
                target=parsed_command.target_object or "N/A",
                destination=parsed_command.destination or "N/A",
                constraints=", ".join(parsed_command.constraints) if parsed_command.constraints else "None",
                confidence=parsed_command.confidence,
                visual_info=visual_info,
                knowledge_info=knowledge_info
            )
            
            # Generate plan
            response = self.planner_llm.invoke(prompt)
            action_plan = response.content
            
            logger.debug(f"Generated action plan: {action_plan[:100]}...")
            return action_plan
            
        except Exception as e:
            logger.error(f"Failed to generate action plan: {e}")
            return f"Error generating action plan: {e}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of all agent modules.
        
        Returns:
            Dictionary with module status information
        """
        status = {
            "language_module": "initialized",
            "rag_module": "initialized",
            "vision_module": "enabled" if self.vision_module else "disabled",
            "knowledge_base_stats": self.rag_module.get_collection_stats()
        }
        return status
    
    def load_knowledge_base(self, file_path: str) -> int:
        """
        Load or reload knowledge base from file.
        
        Args:
            file_path: Path to knowledge base JSON file
        
        Returns:
            Number of entries loaded
        """
        try:
            count = self.rag_module.load_knowledge_from_file(file_path)
            logger.info(f"Loaded {count} knowledge entries")
            return count
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            raise

