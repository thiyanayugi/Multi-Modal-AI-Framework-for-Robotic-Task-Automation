"""RAG Module for knowledge retrieval using ChromaDB and semantic search.

This module provides Retrieval-Augmented Generation (RAG) capabilities for robotic task automation.
It enables the agent to:
- Store and retrieve domain-specific knowledge about manipulation strategies
- Perform semantic search using sentence transformers
- Filter knowledge by metadata categories
- Load knowledge bases from JSON files

The module uses ChromaDB for vector storage and sentence-transformers for embedding generation,
providing fast and accurate semantic retrieval to augment the agent's decision-making.
"""

# Fix for SQLite version issue with ChromaDB
import sys
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Any, Union

# Disable ChromaDB telemetry to avoid Python 3.8 compatibility issues
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGModule:
    """
    Retrieval-Augmented Generation module using ChromaDB.
    
    Provides semantic search over a knowledge base of robotic manipulation
    strategies and procedures using vector embeddings and similarity search.
    
    The module enables:
    - Storage of domain-specific knowledge with metadata
    - Semantic retrieval using natural language queries
    - Batch knowledge loading from JSON files
    - Metadata-based filtering for targeted retrieval
    
    Example:
        >>> rag = RAGModule(collection_name="robot_knowledge")
        >>> # Add knowledge entry
        >>> rag.add_knowledge(
        ...     "When grasping blocks, approach from the top with 50% grip force.",
        ...     metadata={"category": "grasping", "object": "block"}
        ... )
        >>> 
        >>> # Retrieve relevant knowledge
        >>> results = rag.retrieve("how to pick up a block", top_k=3)
        >>> print(results[0]["content"])  # "When grasping blocks, approach from..."
        >>> print(results[0]["score"])  # 0.92
    """
    
    def __init__(
        self,
        collection_name: str = "robotic_knowledge",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAGModule with ChromaDB and sentence transformers.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database (None for in-memory)
            embedding_model: Sentence transformer model for embeddings
        
        Raises:
            RuntimeError: If initialization fails
        """
        try:
            # Initialize embedding model
            # Sentence transformers convert text to dense vectors for semantic search
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            
            # Setup ChromaDB
            # Use persistent storage to maintain knowledge across sessions
            persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            
            # Create persistent client with telemetry disabled for privacy
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,  # Disable telemetry for privacy
                    allow_reset=True  # Allow collection reset for testing
                )
            )
            
            # Get or create collection
            # Collections store embeddings and metadata for efficient retrieval
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Robotic manipulation knowledge base"}
            )
            
            logger.info(f"RAGModule initialized with collection '{collection_name}' "
                       f"({self.collection.count()} documents)")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGModule: {e}")
            raise RuntimeError(f"RAGModule initialization failed: {e}")
    
    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a knowledge entry to the database.
        
        Args:
            content: Text content of the knowledge entry
            metadata: Optional metadata (category, tags, etc.)
            doc_id: Optional document ID (generated if not provided)
        
        Returns:
            Document ID of the added entry
        
        Raises:
            ValueError: If content is empty
            RuntimeError: If addition fails
        """
        try:
            if not content or not content.strip():
                raise ValueError("Content cannot be empty")
            
            # Generate ID if not provided
            # Use UUID for unique, collision-free identifiers
            doc_id = doc_id or str(uuid.uuid4())
            
            # Generate embedding
            # Convert text to dense vector representation for semantic search
            embedding = self.embedding_model.encode(content).tolist()
            
            # Prepare metadata
            # Add content length for potential filtering or analysis
            meta = metadata or {}
            meta["content_length"] = len(content)
            
            # Add to collection
            # Store document with its embedding and metadata in ChromaDB
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[meta]
            )
            
            logger.info(f"Added knowledge entry: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            raise RuntimeError(f"Knowledge addition failed: {e}")
    
    def add_knowledge_batch(
        self,
        entries: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple knowledge entries in batch.
        
        Args:
            entries: List of dicts with 'content', optional 'metadata' and 'id'
        
        Returns:
            List of document IDs
        
        Raises:
            ValueError: If entries is empty or invalid
            RuntimeError: If batch addition fails
        """
        try:
            if not entries:
                raise ValueError("Entries list cannot be empty")
            
            ids = []
            embeddings = []
            documents = []
            embeddings_list = []
            metadatas = []
            
            # Process each entry
            # Generate embeddings and prepare metadata for all entries
            for entry in entries:
                content = entry.get("content")
                if not content or not content.strip():
                    logger.warning("Skipping empty content in batch")
                    continue
                
                # Generate unique ID if not provided
                doc_id = entry.get("id", str(uuid.uuid4()))
                ids.append(doc_id)
                
                # Store document content
                documents.append(content)
                
                # Generate embedding for semantic search
                embedding = self.embedding_model.encode(content).tolist()
                embeddings_list.append(embedding)
                
                # Prepare metadata with content length
                meta = entry.get("metadata", {})
                meta["content_length"] = len(content)
                metadatas.append(meta)
            
            # Batch add to collection
            # Single ChromaDB operation is much faster than individual inserts
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(ids)} knowledge entries in batch")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add knowledge batch: {e}")
            raise RuntimeError(f"Batch addition failed: {e}")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge entries using semantic search.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of dicts with 'content', 'metadata', 'score', and 'id'
        
        Raises:
            ValueError: If query is empty or top_k is invalid
            RuntimeError: If retrieval fails
        """
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if top_k < 1:
                raise ValueError("top_k must be at least 1")
            
            # Generate query embedding
            # Convert text query to vector representation for similarity search
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query collection
            # ChromaDB uses cosine similarity to find most relevant documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),  # Don't request more than available
                where=filter_metadata  # Optional metadata filtering
            )
            
            # Format results
            # Convert ChromaDB response to standardized dictionary format
            retrieved = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    retrieved.append({
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        # Convert distance to similarity score (1 - distance)
                        # Higher score = more similar
                        "score": 1 - results["distances"][0][i] if results["distances"] else 1.0
                    })
            
            logger.info(f"Retrieved {len(retrieved)} documents for query: '{query[:50]}...'")
            return retrieved
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}")
            raise RuntimeError(f"Knowledge retrieval failed: {e}")
    
    def load_knowledge_from_file(self, file_path: str) -> int:
        """
        Load knowledge entries from a JSON file.
        
        Expected JSON format:
        [
            {
                "content": "text content",
                "metadata": {"category": "...", ...},
                "id": "optional-id"
            },
            ...
        ]
        
        Args:
            file_path: Path to JSON file
        
        Returns:
            Number of entries loaded
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON format is invalid
            RuntimeError: If loading fails
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Knowledge base file not found: {file_path}")
            
            # Load JSON file
            # Expected format: list of objects with 'content' and optional 'metadata' fields
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate format
            # Ensure data is a list of knowledge entries
            if not isinstance(data, list):
                raise ValueError("Knowledge base file must contain a JSON array")
            
            # Process entries
            # Convert JSON objects to the format expected by add_knowledge_batch
            entries = []
            for item in data:
                if isinstance(item, dict) and "content" in item:
                    entries.append(item)
                elif isinstance(item, str):
                    # Support simple string arrays for backward compatibility
                    entries.append({"content": item})
                else:
                    logger.warning(f"Skipping invalid entry: {item}")
            
            # Batch add all entries
            # More efficient than adding one at a time
            count = self.add_knowledge_batch(entries)
            logger.info(f"Loaded {count} entries from {file_path}")
            
            return count
            
        except FileNotFoundError:
            # Re-raise file not found errors for caller to handle
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in knowledge base file: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            raise RuntimeError(f"Knowledge base loading failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            # Get collection count
            # Total number of documents in the knowledge base
            count = self.collection.count()
            
            # Get sample documents
            # Retrieve a few entries for preview/debugging
            sample_size = min(5, count)
            if count > 0:
                results = self.collection.peek(limit=sample_size)
                sample_docs = results.get("documents", [])
            else:
                sample_docs = []
            
            # Compile statistics
            # Provide overview of knowledge base contents
            stats = {
                "total_documents": count,
                "collection_name": self.collection.name,
                "sample_documents": sample_docs[:3],  # Show first 3 for brevity
                "embedding_model": str(self.embedding_model)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                "total_documents": 0,
                "error": str(e)
            }
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        
        Warning: This operation cannot be undone.
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"description": "Robotic manipulation knowledge base"}
            )
            logger.warning(f"Cleared collection '{self.collection.name}'")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise RuntimeError(f"Collection clearing failed: {e}")
    
    def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search documents by metadata filters.
        
        Args:
            metadata_filter: Metadata conditions to filter by
            limit: Maximum number of results
        
        Returns:
            List of matching documents
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit
            )
            
            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    documents.append({
                        "id": doc_id,
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })
            
            logger.info(f"Found {len(documents)} documents matching metadata filter")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search by metadata: {e}")
            return []

