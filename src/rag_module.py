"""RAG Module for knowledge retrieval using ChromaDB and semantic search."""

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
    strategies and procedures.
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
            logger.info(f"Loading embedding model: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            
            # Setup ChromaDB
            persist_dir = persist_directory or os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
            
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
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
            doc_id = doc_id or str(uuid.uuid4())
            
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()
            
            # Prepare metadata
            meta = metadata or {}
            meta["content_length"] = len(content)
            
            # Add to collection
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
            metadatas = []
            
            for entry in entries:
                if "content" not in entry:
                    raise ValueError("Each entry must have 'content' field")
                
                content = entry["content"]
                doc_id = entry.get("id", str(uuid.uuid4()))
                metadata = entry.get("metadata", {})

                # Convert list values to comma-separated strings (ChromaDB requirement)
                for key, value in metadata.items():
                    if isinstance(value, list):
                        metadata[key] = ", ".join(str(v) for v in value)

                metadata["content_length"] = len(content)
                
                # Generate embedding
                embedding = self.embedding_model.encode(content).tolist()
                
                ids.append(doc_id)
                embeddings.append(embedding)
                documents.append(content)
                metadatas.append(metadata)
            
            # Add batch to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
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
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=filter_metadata
            )
            
            # Format results
            retrieved = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    retrieved.append({
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
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
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON file must contain a list of entries")
            
            # Add entries in batch
            ids = self.add_knowledge_batch(data)
            
            logger.info(f"Loaded {len(ids)} knowledge entries from {file_path}")
            return len(ids)
            
        except Exception as e:
            logger.error(f"Failed to load knowledge from file: {e}")
            raise RuntimeError(f"Knowledge loading failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            stats = {
                "total_documents": count,
                "collection_name": self.collection.name,
                "metadata": self.collection.metadata
            }
            
            # Get sample metadata if documents exist
            if count > 0:
                sample = self.collection.get(limit=1)
                if sample["metadatas"]:
                    stats["sample_metadata_keys"] = list(sample["metadatas"][0].keys())
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
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

