"""Performance tests for the sentiment analysis engine."""

import time
import pytest
from typing import List
from src.utils.performance import PerformanceMonitor, ProgressTracker, BatchProcessor
from src.models.review import Review
from src.analysis.sentiment_analyzer import VaderSentimentAnalyzer


class TestPerformanceMonitor:
    """Test cases for PerformanceMonitor."""
    
    def test_monitor_operation(self):
        """Test operation monitoring."""
        monitor = PerformanceMonitor()
        
        with monitor.monitor_operation("test_operation", 100) as metrics:
            # Simulate some work
            time.sleep(0.1)
            # Metrics object should be populated during execution
            assert metrics.operation_name == "test_operation"
            assert metrics.items_processed == 100
        
        # After context exit, metrics should be complete
        assert metrics.duration >= 0.1
        assert metrics.end_time > metrics.start_time
        assert len(monitor.metrics_history) == 1
    
    def test_get_operation_stats(self):
        """Test operation statistics retrieval."""
        monitor = PerformanceMonitor()
        
        # Run same operation multiple times
        for i in range(3):
            with monitor.monitor_operation("repeated_op", 10):
                time.sleep(0.05)
        
        stats = monitor.get_operation_stats("repeated_op")
        
        assert stats["operation_name"] == "repeated_op"
        assert stats["execution_count"] == 3
        assert "duration" in stats
        assert "memory_usage" in stats
        assert "processing_rate" in stats
    
    def test_get_overall_stats(self):
        """Test overall statistics."""
        monitor = PerformanceMonitor()
        
        with monitor.monitor_operation("op1", 50):
            time.sleep(0.02)
        
        with monitor.monitor_operation("op2", 30):
            time.sleep(0.03)
        
        stats = monitor.get_overall_stats()
        
        assert stats["total_operations"] == 2
        assert stats["unique_operations"] == 2
        assert stats["total_items_processed"] == 80
        assert stats["total_duration"] >= 0.05


class TestProgressTracker:
    """Test cases for ProgressTracker."""
    
    def test_progress_tracking(self):
        """Test basic progress tracking."""
        tracker = ProgressTracker(100, "Test Operation")
        
        # Update progress
        tracker.update(25)
        assert tracker.processed_items == 25
        
        tracker.update(25)
        assert tracker.processed_items == 50
        
        # Finish and get stats
        stats = tracker.finish()
        
        assert stats["total_items"] == 100
        assert stats["processed_items"] == 50
        assert stats["completion_percentage"] == 50.0
        assert "duration" in stats
        assert "average_rate" in stats
    
    def test_progress_milestones(self):
        """Test progress milestone reporting."""
        tracker = ProgressTracker(100, "Milestone Test")
        
        # Should trigger milestone reports at 10%, 25%, etc.
        tracker.update(10)  # 10%
        tracker.update(15)  # 25%
        tracker.update(25)  # 50%
        
        # Check that milestones were recorded
        assert 10 in tracker.reported_milestones
        assert 25 in tracker.reported_milestones


class TestBatchProcessor:
    """Test cases for BatchProcessor."""
    
    def test_batch_processing(self):
        """Test basic batch processing."""
        processor = BatchProcessor(batch_size=10)
        
        # Create test data
        items = list(range(25))  # 25 items
        
        def simple_processor(batch):
            return [x * 2 for x in batch]
        
        results = processor.process_in_batches(items, simple_processor)
        
        assert len(results) == 25
        assert results[0] == 0
        assert results[24] == 48
    
    def test_empty_batch_processing(self):
        """Test batch processing with empty input."""
        processor = BatchProcessor()
        
        def dummy_processor(batch):
            return batch
        
        results = processor.process_in_batches([], dummy_processor)
        assert results == []
    
    def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        processor = BatchProcessor(batch_size=100)
        
        # Test with different dataset sizes
        batch_size_small = processor._calculate_optimal_batch_size(50)
        batch_size_large = processor._calculate_optimal_batch_size(10000)
        
        assert batch_size_small <= 50  # Should not exceed total items
        assert batch_size_large >= 1   # Should be at least 1
        assert batch_size_large <= 1000  # Should not exceed absolute maximum


class TestSentimentAnalysisPerformance:
    """Performance tests for sentiment analysis components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = VaderSentimentAnalyzer()
    
    def create_test_reviews(self, count: int) -> List[Review]:
        """Create test reviews for performance testing."""
        reviews = []
        sentiments = [
            "Excellent produit, je le recommande !",
            "Service client horrible, très déçu.",
            "Produit correct, sans plus.",
            "Fantastique ! Vraiment génial.",
            "Qualité décevante pour le prix."
        ]
        
        for i in range(count):
            review = Review(
                review_id=f"PERF{i:04d}",
                review_text=sentiments[i % len(sentiments)]
            )
            reviews.append(review)
        
        return reviews
    
    def test_small_dataset_performance(self):
        """Test performance with small dataset (< 100 reviews)."""
        reviews = self.create_test_reviews(50)
        
        start_time = time.time()
        results = self.analyzer.analyze_sentiment(reviews)
        duration = time.time() - start_time
        
        assert len(results) == 50
        assert all(r.sentiment_label in ["Positive", "Negative", "Neutral"] for r in results)
        assert duration < 5.0  # Should complete in under 5 seconds
    
    def test_medium_dataset_performance(self):
        """Test performance with medium dataset (100-500 reviews)."""
        reviews = self.create_test_reviews(200)
        
        start_time = time.time()
        results = self.analyzer.analyze_sentiment(reviews)
        duration = time.time() - start_time
        
        assert len(results) == 200
        assert duration < 15.0  # Should complete in under 15 seconds
        
        # Check processing rate
        rate = len(results) / duration
        assert rate > 10  # Should process at least 10 reviews per second
    
    def test_large_dataset_performance(self):
        """Test performance with large dataset (500+ reviews)."""
        reviews = self.create_test_reviews(1000)
        
        start_time = time.time()
        results = self.analyzer.analyze_sentiment(reviews)
        duration = time.time() - start_time
        
        assert len(results) == 1000
        assert duration < 60.0  # Should complete in under 1 minute
        
        # Check processing rate
        rate = len(results) / duration
        assert rate > 15  # Should process at least 15 reviews per second
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable during processing."""
        import psutil
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process multiple batches
        for batch_num in range(5):
            reviews = self.create_test_reviews(100)
            results = self.analyzer.analyze_sentiment(reviews)
            
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable (< 100MB per batch)
            assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f}MB"
    
    @pytest.mark.slow
    def test_stress_test(self):
        """Stress test with very large dataset."""
        reviews = self.create_test_reviews(5000)
        
        monitor = PerformanceMonitor()
        
        with monitor.monitor_operation("stress_test", len(reviews)) as metrics:
            results = self.analyzer.analyze_sentiment(reviews)
        
        assert len(results) == 5000
        assert metrics.duration < 300  # Should complete in under 5 minutes
        assert metrics.items_per_second > 10  # Minimum processing rate
        
        # Check that all reviews were processed
        assert all(r.sentiment_score is not None for r in results)
        assert all(r.sentiment_label is not None for r in results)


class TestIntegrationPerformance:
    """Integration performance tests."""
    
    def test_end_to_end_performance(self):
        """Test end-to-end pipeline performance."""
        from src.engine import SentimentAnalysisEngine
        import tempfile
        import json
        
        # Create test data
        test_reviews = []
        for i in range(500):
            test_reviews.append({
                "review_id": f"E2E{i:04d}",
                "review_text": f"Test review number {i} with sentiment."
            })
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input file
            input_file = f"{temp_dir}/test_reviews.json"
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(test_reviews, f)
            
            # Run analysis
            engine = SentimentAnalysisEngine()
            
            start_time = time.time()
            results = engine.analyze_reviews(input_file, temp_dir)
            duration = time.time() - start_time
            
            # Verify results
            assert results["status"] == "completed"
            assert results["statistics"]["total_reviews"] == 500
            assert duration < 30.0  # Should complete in under 30 seconds
            
            # Check performance metrics
            if "performance_metrics" in results:
                perf_metrics = results["performance_metrics"]
                # Total items processed includes all pipeline steps (4 steps * 500 items = 2000)
                assert perf_metrics["total_items_processed"] >= 500
                assert perf_metrics["overall_rate"] > 0  # Should have positive processing rate