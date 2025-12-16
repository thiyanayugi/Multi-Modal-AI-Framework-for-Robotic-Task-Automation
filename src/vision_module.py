"""Vision Module for CLIP-based image understanding and object detection."""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VisionModule:
    """
    CLIP-based vision module for robotic task automation.
    
    Provides image encoding, object detection, and visual context extraction
    using OpenAI's CLIP model.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None):
        """
        Initialize the VisionModule with CLIP model.
        
        Args:
            model_name: HuggingFace model identifier for CLIP
            device: Device to run model on ('cuda', 'cpu', or None for auto-detect)
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Initializing VisionModule with {model_name} on {self.device}")
            
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            
            logger.info("VisionModule initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VisionModule: {e}")
            raise RuntimeError(f"VisionModule initialization failed: {e}")
    
    def encode_image(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Encode an image into CLIP embeddings.
        
        Args:
            image: Image as file path, PIL Image, or numpy array
        
        Returns:
            numpy array of image embeddings (shape: [512])
        
        Raises:
            ValueError: If image format is invalid
            RuntimeError: If encoding fails
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Process and encode
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embeddings = image_features.cpu().numpy().flatten()
            logger.debug(f"Encoded image to embeddings of shape {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            raise RuntimeError(f"Image encoding failed: {e}")
    
    def find_objects(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text_queries: List[str],
        threshold: float = 0.2
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Find objects in an image matching text queries using CLIP.
        
        Args:
            image: Image as file path, PIL Image, or numpy array
            text_queries: List of text descriptions to match against
            threshold: Minimum similarity score (0-1) to consider a match
        
        Returns:
            List of dicts with 'object', 'confidence', and 'description' keys,
            sorted by confidence (highest first)
        
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If object detection fails
        """
        try:
            if not text_queries:
                raise ValueError("text_queries cannot be empty")
            
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Process inputs
            inputs = self.processor(
                text=text_queries,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Get similarity scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
            # Filter and format results
            results = []
            for query, confidence in zip(text_queries, probs):
                if confidence >= threshold:
                    results.append({
                        "object": query.split()[0] if query else "unknown",
                        "description": query,
                        "confidence": float(confidence)
                    })
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Found {len(results)} objects matching queries")
            return results
            
        except Exception as e:
            logger.error(f"Failed to find objects: {e}")
            raise RuntimeError(f"Object detection failed: {e}")
    
    def get_visual_context(
        self,
        image: Union[str, Image.Image, np.ndarray],
        context_queries: Optional[List[str]] = None
    ) -> Dict[str, Union[List[Dict], str]]:
        """
        Extract comprehensive visual context from workspace image.
        
        Args:
            image: Image as file path, PIL Image, or numpy array
            context_queries: Optional list of specific elements to detect.
                           If None, uses default workspace queries.
        
        Returns:
            Dictionary containing:
                - 'detected_objects': List of detected objects with confidence
                - 'workspace_description': Text description of the scene
                - 'embeddings': Image embeddings for further processing
        
        Raises:
            RuntimeError: If context extraction fails
        """
        try:
            # Default workspace queries
            if context_queries is None:
                context_queries = [
                    "red block on table",
                    "blue block on table",
                    "green block on table",
                    "yellow block on table",
                    "empty workspace",
                    "cluttered workspace",
                    "robot gripper",
                    "target location",
                    "obstacle on table",
                    "organized blocks"
                ]
            
            # Get image embeddings
            embeddings = self.encode_image(image)
            
            # Detect objects
            detected_objects = self.find_objects(image, context_queries, threshold=0.15)
            
            # Generate workspace description
            if detected_objects:
                top_matches = detected_objects[:3]
                descriptions = [obj["description"] for obj in top_matches]
                workspace_description = f"Workspace contains: {', '.join(descriptions)}"
            else:
                workspace_description = "Workspace scene detected but no specific objects identified"
            
            context = {
                "detected_objects": detected_objects,
                "workspace_description": workspace_description,
                "embeddings": embeddings.tolist(),
                "num_objects": len(detected_objects)
            }
            
            logger.info(f"Extracted visual context: {context['workspace_description']}")
            return context
            
        except Exception as e:
            logger.error(f"Failed to extract visual context: {e}")
            raise RuntimeError(f"Visual context extraction failed: {e}")
    
    def compare_images(
        self,
        image1: Union[str, Image.Image, np.ndarray],
        image2: Union[str, Image.Image, np.ndarray]
    ) -> float:
        """
        Compare two images using cosine similarity of their CLIP embeddings.
        
        Args:
            image1: First image
            image2: Second image
        
        Returns:
            Similarity score between 0 and 1 (1 = identical)
        
        Raises:
            RuntimeError: If comparison fails
        """
        try:
            emb1 = self.encode_image(image1)
            emb2 = self.encode_image(image2)
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            logger.debug(f"Image similarity: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compare images: {e}")
            raise RuntimeError(f"Image comparison failed: {e}")

