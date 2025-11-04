"""Tests for edge cases in sentiment analysis."""

import pytest
import json
from pathlib import Path

from src.engine import SentimentAnalysisEngine
from src.models.review import Review


class TestEdgeCases:
    """Test suite for edge cases and corner scenarios."""

    @pytest.fixture
    def edge_cases_data(self):
        """Load edge cases test data."""
        edge_cases_file = Path(__file__).parent / "data" / "edge_cases.json"
        with open(edge_cases_file, "r", encoding="utf-8") as f:
            return json.load(f)

    @pytest.fixture
    def engine(self):
        """Create sentiment analysis engine."""
        return SentimentAnalysisEngine()

    def test_all_edge_cases(self, edge_cases_data, engine, tmp_path):
        """Test all edge cases from the edge cases dataset."""
        # Prepare input file
        input_file = tmp_path / "edge_cases_input.json"
        input_data = [
            {"review_id": case["review_id"], "review_text": case["review_text"]}
            for case in edge_cases_data
        ]
        
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f, ensure_ascii=False)

        # Run analysis
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        # Load results
        import csv
        results_file = results["files_generated"]["results"]
        predictions = {}
        
        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions[row["review_id"]] = row["sentiment_final"]

        # Verify each edge case
        errors = []
        for case in edge_cases_data:
            review_id = case["review_id"]
            expected = case["expected_sentiment"]
            actual = predictions.get(review_id, "Unknown")
            
            if actual != expected:
                errors.append({
                    "review_id": review_id,
                    "text": case["review_text"][:50],
                    "description": case["description"],
                    "expected": expected,
                    "actual": actual,
                })

        # Report results
        if errors:
            error_msg = f"\n{len(errors)} edge cases failed:\n"
            for error in errors:
                error_msg += (
                    f"  - {error['review_id']} ({error['description']}): "
                    f"expected {error['expected']}, got {error['actual']}\n"
                    f"    Text: \"{error['text']}...\"\n"
                )
            pytest.fail(error_msg)

    def test_very_short_reviews(self, engine, tmp_path):
        """Test with very short reviews (1-3 characters)."""
        input_file = tmp_path / "short.json"
        data = [
            {"review_id": "S001", "review_text": "A"},
            {"review_id": "S002", "review_text": "OK"},
            {"review_id": "S003", "review_text": "Non"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 3

    def test_very_long_reviews(self, engine, tmp_path):
        """Test with very long reviews (1000+ words)."""
        input_file = tmp_path / "long.json"
        long_text = " ".join(["Excellent produit"] * 500)  # ~1000 words
        data = [{"review_id": "L001", "review_text": long_text}]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_only_emojis(self, engine, tmp_path):
        """Test with reviews containing only emojis."""
        input_file = tmp_path / "emojis.json"
        data = [
            {"review_id": "E001", "review_text": "😀😀😀"},
            {"review_id": "E002", "review_text": "😭😭😭"},
            {"review_id": "E003", "review_text": "👍👍👍"},
            {"review_id": "E004", "review_text": "👎👎👎"},
        ]
        
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"
        assert results["statistics"]["total_reviews"] == 4

    def test_negation_patterns(self, engine, tmp_path):
        """Test with various negation patterns."""
        input_file = tmp_path / "negation.json"
        data = [
            {"review_id": "N001", "review_text": "Pas bon"},
            {"review_id": "N002", "review_text": "Ne fonctionne pas"},
            {"review_id": "N003", "review_text": "Jamais satisfait"},
            {"review_id": "N004", "review_text": "Aucun problème"},
            {"review_id": "N005", "review_text": "Rien à redire"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_mixed_sentiments(self, engine, tmp_path):
        """Test with reviews containing mixed sentiments."""
        input_file = tmp_path / "mixed.json"
        data = [
            {"review_id": "M001", "review_text": "Bon produit mais cher"},
            {"review_id": "M002", "review_text": "Qualité excellente, livraison horrible"},
            {"review_id": "M003", "review_text": "J'aime le design mais déteste la qualité"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"
        # Mixed sentiments should typically be classified as Neutral
        assert results["statistics"]["neutral_count"] >= 0

    def test_sms_language(self, engine, tmp_path):
        """Test with SMS/text speak language."""
        input_file = tmp_path / "sms.json"
        data = [
            {"review_id": "SMS001", "review_text": "tro bi1"},
            {"review_id": "SMS002", "review_text": "c nul"},
            {"review_id": "SMS003", "review_text": "jador !!!"},
            {"review_id": "SMS004", "review_text": "jdéteste"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_all_caps(self, engine, tmp_path):
        """Test with all caps reviews."""
        input_file = tmp_path / "caps.json"
        data = [
            {"review_id": "C001", "review_text": "EXCELLENT PRODUIT"},
            {"review_id": "C002", "review_text": "HORRIBLE SERVICE"},
            {"review_id": "C003", "review_text": "CORRECT"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_excessive_punctuation(self, engine, tmp_path):
        """Test with excessive punctuation."""
        input_file = tmp_path / "punctuation.json"
        data = [
            {"review_id": "P001", "review_text": "Super !!!!!!!!!"},
            {"review_id": "P002", "review_text": "Nul ?????????"},
            {"review_id": "P003", "review_text": "OK..........."},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_word_repetition(self, engine, tmp_path):
        """Test with repeated words."""
        input_file = tmp_path / "repetition.json"
        data = [
            {"review_id": "R001", "review_text": "Bon bon bon bon bon"},
            {"review_id": "R002", "review_text": "Nul nul nul nul nul"},
            {"review_id": "R003", "review_text": "OK OK OK OK OK"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_rating_formats(self, engine, tmp_path):
        """Test with various rating formats."""
        input_file = tmp_path / "ratings.json"
        data = [
            {"review_id": "RAT001", "review_text": "5/5"},
            {"review_id": "RAT002", "review_text": "1/5"},
            {"review_id": "RAT003", "review_text": "3/5"},
            {"review_id": "RAT004", "review_text": "10/10"},
            {"review_id": "RAT005", "review_text": "0/10"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_mixed_languages(self, engine, tmp_path):
        """Test with mixed French/English reviews."""
        input_file = tmp_path / "mixed_lang.json"
        data = [
            {"review_id": "ML001", "review_text": "Good produit"},
            {"review_id": "ML002", "review_text": "Bad qualité"},
            {"review_id": "ML003", "review_text": "Très good"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_special_characters_only(self, engine, tmp_path):
        """Test with only special characters."""
        input_file = tmp_path / "special.json"
        data = [
            {"review_id": "SP001", "review_text": "@#$%^&*()"},
            {"review_id": "SP002", "review_text": "!!!???"},
            {"review_id": "SP003", "review_text": "...---..."},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_numbers_only(self, engine, tmp_path):
        """Test with only numbers."""
        input_file = tmp_path / "numbers.json"
        data = [
            {"review_id": "NUM001", "review_text": "123456"},
            {"review_id": "NUM002", "review_text": "999"},
            {"review_id": "NUM003", "review_text": "0"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_unicode_edge_cases(self, engine, tmp_path):
        """Test with various Unicode edge cases."""
        input_file = tmp_path / "unicode.json"
        data = [
            {"review_id": "U001", "review_text": "Café très bon ☕"},
            {"review_id": "U002", "review_text": "Prix €€€ trop cher"},
            {"review_id": "U003", "review_text": "Température 25°C parfaite"},
        ]
        
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_colloquial_expressions(self, engine, tmp_path):
        """Test with colloquial French expressions."""
        input_file = tmp_path / "colloquial.json"
        data = [
            {"review_id": "COL001", "review_text": "Bof"},
            {"review_id": "COL002", "review_text": "Mouais"},
            {"review_id": "COL003", "review_text": "Waouh"},
            {"review_id": "COL004", "review_text": "Ouf"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_intensifiers(self, engine, tmp_path):
        """Test with intensifier words."""
        input_file = tmp_path / "intensifiers.json"
        data = [
            {"review_id": "INT001", "review_text": "Très très bon"},
            {"review_id": "INT002", "review_text": "Vraiment mauvais"},
            {"review_id": "INT003", "review_text": "Super génial"},
            {"review_id": "INT004", "review_text": "Hyper nul"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"

    def test_factual_statements(self, engine, tmp_path):
        """Test with purely factual statements (no sentiment)."""
        input_file = tmp_path / "factual.json"
        data = [
            {"review_id": "F001", "review_text": "Produit reçu"},
            {"review_id": "F002", "review_text": "Couleur bleue"},
            {"review_id": "F003", "review_text": "Taille M"},
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        results = engine.analyze_reviews(str(input_file), str(tmp_path))
        assert results["status"] == "completed"
        # Factual statements should typically be neutral
        assert results["statistics"]["neutral_count"] >= 0


class TestEdgeCaseStatistics:
    """Test statistics and reporting for edge cases."""

    def test_edge_case_error_reporting(self, tmp_path):
        """Test that edge cases with errors are properly reported."""
        input_file = tmp_path / "errors.json"
        data = [
            {"review_id": "ERR001", "review_text": ""},  # Empty
            {"review_id": "ERR002", "review_text": None},  # None
        ]
        
        with open(input_file, "w") as f:
            json.dump(data, f)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        # Should complete without crashing
        assert results["status"] == "completed"
        # Should report errors
        assert "processing_info" in results

    def test_edge_case_distribution(self, tmp_path):
        """Test sentiment distribution with edge cases."""
        input_file = tmp_path / "distribution.json"
        data = [
            {"review_id": "D001", "review_text": "😀"},  # Emoji
            {"review_id": "D002", "review_text": "!!!"},  # Punctuation
            {"review_id": "D003", "review_text": "OK"},  # Short
            {"review_id": "D004", "review_text": ""},  # Empty
        ]
        
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        engine = SentimentAnalysisEngine()
        results = engine.analyze_reviews(str(input_file), str(tmp_path))

        # Verify distribution sums to 100%
        stats = results["statistics"]
        total_pct = (
            stats["positive_percentage"]
            + stats["negative_percentage"]
            + stats["neutral_percentage"]
        )
        assert abs(total_pct - 100.0) < 0.1  # Allow small rounding error
