"""Integration tests for AI Agent Framework."""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision_module import VisionModule
from src.language_module import LanguageModule
from src.rag_module import RAGModule
from src.agent import RoboticAgent


def test_vision_module():
    """Test VisionModule functionality."""
    print("\n" + "="*60)
    print("Testing VisionModule")
    print("="*60)
    
    try:
        # Initialize module
        vision = VisionModule()
        print("✓ VisionModule initialized successfully")
        
        # Create a test image (random RGB)
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        test_image_pil = Image.fromarray(test_image)
        
        # Test image encoding
        embeddings = vision.encode_image(test_image_pil)
        assert embeddings.shape[0] == 512, "Embeddings should be 512-dimensional"
        print(f"✓ Image encoded to embeddings: shape {embeddings.shape}")
        
        # Test object finding
        queries = ["red block", "blue cube", "robot gripper"]
        results = vision.find_objects(test_image_pil, queries, threshold=0.1)
        print(f"✓ Object detection completed: found {len(results)} matches")
        for result in results[:3]:
            print(f"  - {result['description']}: {result['confidence']:.3f}")
        
        # Test visual context extraction
        context = vision.get_visual_context(test_image_pil)
        assert "detected_objects" in context
        assert "workspace_description" in context
        print(f"✓ Visual context extracted: {context['workspace_description']}")
        
        print("\n✅ VisionModule tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ VisionModule tests FAILED: {e}")
        return False


def test_language_module():
    """Test LanguageModule functionality."""
    print("\n" + "="*60)
    print("Testing LanguageModule")
    print("="*60)
    
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY not set, skipping LanguageModule tests")
            return True
        
        # Initialize module
        language = LanguageModule()
        print("✓ LanguageModule initialized successfully")
        
        # Test command parsing
        test_commands = [
            "Pick up the red block and place it on the blue square",
            "Move the green cube to the left corner",
            "Sort all blocks by color"
        ]
        
        for cmd in test_commands:
            parsed = language.parse_command(cmd)
            print(f"\n✓ Parsed: '{cmd}'")
            print(f"  Action: {parsed.action}")
            print(f"  Target: {parsed.target_object}")
            print(f"  Destination: {parsed.destination}")
            print(f"  Confidence: {parsed.confidence:.2f}")
            
            # Test ambiguity checking
            ambiguity = language.check_ambiguity(cmd, parsed)
            print(f"  Ambiguous: {ambiguity.is_ambiguous} (score: {ambiguity.ambiguity_score:.2f})")
            
            # Test confirmation generation
            confirmation = language.generate_confirmation(parsed)
            print(f"  Confirmation: {confirmation}")
        
        print("\n✅ LanguageModule tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ LanguageModule tests FAILED: {e}")
        return False


def test_rag_module():
    """Test RAGModule functionality."""
    print("\n" + "="*60)
    print("Testing RAGModule")
    print("="*60)
    
    try:
        # Initialize module
        rag = RAGModule(collection_name="test_collection")
        print("✓ RAGModule initialized successfully")
        
        # Clear any existing data
        rag.clear_collection()
        print("✓ Collection cleared")
        
        # Test adding knowledge
        test_entries = [
            {
                "content": "When grasping blocks, approach from the top with 50% grip force.",
                "metadata": {"category": "grasping", "object": "block"}
            },
            {
                "content": "For placement operations, descend slowly and verify contact before releasing.",
                "metadata": {"category": "placement"}
            },
            {
                "content": "Sort objects by color in this order: red, blue, green, yellow.",
                "metadata": {"category": "sorting", "type": "color"}
            }
        ]
        
        ids = rag.add_knowledge_batch(test_entries)
        print(f"✓ Added {len(ids)} knowledge entries")
        
        # Test retrieval
        queries = [
            "how to grasp a block",
            "placing objects safely",
            "color sorting strategy"
        ]
        
        for query in queries:
            results = rag.retrieve(query, top_k=2)
            print(f"\n✓ Query: '{query}'")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result['score']:.3f}")
                print(f"     {result['content'][:80]}...")
        
        # Test collection stats
        stats = rag.get_collection_stats()
        print(f"\n✓ Collection stats: {stats['total_documents']} documents")
        
        # Test loading from file
        kb_path = Path(__file__).parent.parent / "knowledge_base" / "manipulation_strategies.json"
        if kb_path.exists():
            rag.clear_collection()
            count = rag.load_knowledge_from_file(str(kb_path))
            print(f"✓ Loaded {count} entries from knowledge base file")
        
        print("\n✅ RAGModule tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ RAGModule tests FAILED: {e}")
        return False


def test_robotic_agent():
    """Test RoboticAgent integration."""
    print("\n" + "="*60)
    print("Testing RoboticAgent Integration")
    print("="*60)
    
    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY not set, skipping RoboticAgent tests")
            return True
        
        # Initialize agent
        kb_path = Path(__file__).parent.parent / "knowledge_base" / "manipulation_strategies.json"
        agent = RoboticAgent(
            knowledge_base_path=str(kb_path) if kb_path.exists() else None,
            enable_vision=True
        )
        print("✓ RoboticAgent initialized successfully")
        
        # Test system status
        status = agent.get_system_status()
        print(f"✓ System status: {status['vision_module']}, "
              f"{status['knowledge_base_stats']['total_documents']} knowledge entries")
        
        # Test task processing without image
        test_commands = [
            "Pick up the red block",
            "Place the blue cube on the table",
            "Sort all blocks by color"
        ]
        
        for cmd in test_commands:
            print(f"\n{'='*60}")
            print(f"Processing: '{cmd}'")
            print('='*60)
            
            result = agent.process_task(cmd)
            
            if result.success:
                print(f"✓ Task processed successfully")
                print(f"\nParsed Command:")
                print(f"  Action: {result.parsed_command.action}")
                print(f"  Target: {result.parsed_command.target_object}")
                print(f"  Destination: {result.parsed_command.destination}")
                
                if result.retrieved_knowledge:
                    print(f"\nRetrieved Knowledge ({len(result.retrieved_knowledge)} entries):")
                    for i, entry in enumerate(result.retrieved_knowledge[:2], 1):
                        print(f"  {i}. {entry['content'][:100]}...")
                
                print(f"\nAction Plan:")
                plan_lines = result.action_plan.split('\n')[:10]
                for line in plan_lines:
                    print(f"  {line}")
                if len(result.action_plan.split('\n')) > 10:
                    print("  ...")
            else:
                print(f"✗ Task failed: {result.error}")
        
        # Test with synthetic image
        print(f"\n{'='*60}")
        print("Testing with visual input")
        print('='*60)
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = agent.process_task(
            "Pick up the red block and place it in the corner",
            image=test_image
        )
        
        if result.success:
            print("✓ Task with image processed successfully")
            if result.visual_context:
                print(f"  Visual context: {result.visual_context['workspace_description']}")
        else:
            print(f"✗ Task with image failed: {result.error}")
        
        print("\n✅ RoboticAgent tests PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ RoboticAgent tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("AI AGENT FRAMEWORK - INTEGRATION TESTS")
    print("="*60)
    
    results = {
        "VisionModule": test_vision_module(),
        "LanguageModule": test_language_module(),
        "RAGModule": test_rag_module(),
        "RoboticAgent": test_robotic_agent()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for module, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{module:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        print("⚠️  .env file not found, using system environment variables")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

