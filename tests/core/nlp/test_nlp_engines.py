#!/usr/bin/env python3
# tests/core/nlp/test_nlp_engines.py

import pytest
import logging
import os
from unittest.mock import patch, MagicMock

# Add the project root to Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from Redactify.core.nlp.engine_factory import NlpEngineFactory
from Redactify.core.nlp.base_engine import NlpEngine

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

# Sample text for testing
SAMPLE_TEXT = "Hello, my name is John Smith. My email is john.smith@example.com and phone is 555-123-4567."

class TestNlpEngines:
    """Tests for the NLP engines in Redactify."""
    
    def test_engine_factory_default(self):
        """Test that the factory creates a default Spacy engine."""
        config = {}
        engine = NlpEngineFactory.create_engine(config)
        
        # Skip test if dependencies aren't installed
        if engine is None:
            pytest.skip("Spacy dependencies not installed")
            
        assert engine is not None
        assert hasattr(engine, 'process')
        assert hasattr(engine, 'get_tokens')
        assert hasattr(engine, 'extract_entities')
        
    @pytest.mark.parametrize("engine_name", ["spacy", "gliner"])
    def test_engine_factory_specific(self, engine_name):
        """Test that the factory creates specified engines."""
        config = {"engine_name": engine_name}
        
        # For Spacy, use the smallest model for faster tests
        if engine_name == "spacy":
            config["spacy_model"] = "en_core_web_sm"
            
        engine = NlpEngineFactory.create_engine(config)
        
        # Skip test if dependencies aren't installed
        if engine is None:
            pytest.skip(f"{engine_name} dependencies not installed")
            
        assert engine is not None
        assert hasattr(engine, 'process')
        assert hasattr(engine, 'get_tokens')
        assert hasattr(engine, 'extract_entities')
        
        # Process text and verify output structure
        result = engine.process(SAMPLE_TEXT)
        assert "tokens" in result
        assert "entities" in result
        assert len(result["tokens"]) > 0
        
    def test_clean_pdf_text(self):
        """Test the text cleaning functionality."""
        # Text with common PDF extraction issues
        pdf_text = "This doc-\nument contains hyphen-\nated words and ran-\ndom line\nbreaks."
        
        # Create a simple engine instance
        config = {"engine_name": "spacy", "spacy_model": "en_core_web_sm"}
        engine = NlpEngineFactory.create_engine(config)
        
        # Skip test if dependencies aren't installed
        if engine is None:
            pytest.skip("Spacy dependencies not installed")
        
        # Clean the text
        cleaned = engine.clean_pdf_text(pdf_text)
        
        # Verify hyphenation handling
        assert "document" in cleaned
        assert "hyphenated" in cleaned
        assert "random" in cleaned
        assert "doc-ument" not in cleaned
        
    @pytest.mark.parametrize("engine_name", ["spacy", "gliner"])
    def test_entity_detection(self, engine_name):
        """Test that engines can detect common entities."""
        config = {"engine_name": engine_name}
        
        # For Spacy, use the smallest model for faster tests
        if engine_name == "spacy":
            config["spacy_model"] = "en_core_web_sm"
            
        engine = NlpEngineFactory.create_engine(config)
        
        # Skip test if dependencies aren't installed
        if engine is None:
            pytest.skip(f"{engine_name} dependencies not installed")
        
        # Process text with known entities
        result = engine.process(SAMPLE_TEXT)
        
        # There should be entities detected
        assert len(result["entities"]) > 0
        
        # Check basic entity structure
        for entity in result["entities"]:
            assert "text" in entity
            assert "start" in entity
            assert "end" in entity
            assert "label" in entity
            
            # Entity bounds should make sense
            assert entity["start"] >= 0
            assert entity["end"] <= len(SAMPLE_TEXT)
            assert entity["end"] > entity["start"]
            
            # Entity text should match the slice from the original text
            assert entity["text"] == SAMPLE_TEXT[entity["start"]:entity["end"]]
            
    def test_spacy_fallback(self):
        """Test that Spacy falls back to smaller model if specified model fails."""
        # Mock failing to load the specified model but succeeding with fallback
        with patch('spacy.load') as mock_load:
            # First call raises error, second call succeeds
            mock_load.side_effect = [ImportError("Test error"), MagicMock()]
            
            from Redactify.core.nlp.spacy_engine import SpacyEngine
            engine = SpacyEngine(model_name="en_core_web_trf")
            
            # Should have fallen back to en_core_web_sm
            assert mock_load.call_count == 2
            mock_load.assert_any_call("en_core_web_sm")
            
    @pytest.mark.parametrize("engine_name,entity_count_min", [
        ("spacy", 1),  # Spacy should find at least 1
        ("gliner", 2)  # GLiNER should find more entities (email + phone)
    ])
    def test_entity_count_comparison(self, engine_name, entity_count_min):
        """Compare entity detection between engines."""
        config = {"engine_name": engine_name}
        engine = NlpEngineFactory.create_engine(config)
        
        # Skip test if dependencies aren't installed
        if engine is None:
            pytest.skip(f"{engine_name} dependencies not installed")
            
        # Process text with several entities
        text = "Jane Doe works at ACME Inc. Contact her at jane.doe@acme.com or 800-555-1234."
        result = engine.process(text)
        
        # Should find minimum number of entities
        assert len(result["entities"]) >= entity_count_min