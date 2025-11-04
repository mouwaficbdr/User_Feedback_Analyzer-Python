"""Threshold optimization for sentiment classification."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from ..engine import SentimentAnalysisEngine


class ThresholdOptimizer:
    """
    Optimizes sentiment classification thresholds using grid search.
    
    Finds optimal positive and negative thresholds by exhaustive search
    over a parameter grid, evaluating performance on a validation dataset.
    """

    def __init__(self, engine: SentimentAnalysisEngine = None):
        """
        Initialize threshold optimizer.
        
        Args:
            engine: SentimentAnalysisEngine instance to optimize
        """
        self.engine = engine or SentimentAnalysisEngine()
        self.logger = logging.getLogger(__name__)

    def find_optimal_thresholds(
        self,
        validation_file: str,
        metric: str = "f1_score",
        positive_range: Tuple[float, float, float] = (0.01, 0.3, 0.01),
        negative_range: Tuple[float, float, float] = (-0.3, -0.01, 0.01),
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        """
        Find optimal thresholds using grid search.
        
        Args:
            validation_file: Path to validation dataset with expected sentiments
            metric: Metric to optimize ('f1_score', 'accuracy', 'balanced_accuracy')
            positive_range: (start, stop, step) for positive threshold
            negative_range: (start, stop, step) for negative threshold
            output_dir: Directory to save optimization results
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(
            f"Starting threshold optimization with metric: {metric}"
        )
        
        # Load validation data
        validation_data = self._load_validation_data(validation_file)
        
        # Generate parameter grid
        positive_thresholds = np.arange(*positive_range)
        negative_thresholds = np.arange(*negative_range)
        
        total_combinations = len(positive_thresholds) * len(negative_thresholds)
        self.logger.info(
            f"Testing {total_combinations} threshold combinations"
        )
        
        # Grid search
        results = []
        best_score = -np.inf
        best_thresholds = (0.05, -0.05)
        
        for i, pos_thresh in enumerate(positive_thresholds):
            for neg_thresh in negative_thresholds:
                # Skip invalid combinations
                if pos_thresh <= neg_thresh:
                    continue
                
                # Evaluate this threshold combination
                score, metrics = self._evaluate_thresholds(
                    validation_data,
                    pos_thresh,
                    neg_thresh,
                    metric,
                    output_dir,
                )
                
                results.append({
                    "positive_threshold": float(pos_thresh),
                    "negative_threshold": float(neg_thresh),
                    "score": float(score),
                    "accuracy": float(metrics["accuracy"]),
                    "f1_score": float(metrics["f1_score"]),
                    "balanced_accuracy": float(metrics.get("balanced_accuracy", 0)),
                })
                
                if score > best_score:
                    best_score = score
                    best_thresholds = (pos_thresh, neg_thresh)
                    self.logger.info(
                        f"New best: pos={pos_thresh:.3f}, neg={neg_thresh:.3f}, "
                        f"{metric}={score:.4f}"
                    )
            
            # Progress update
            if (i + 1) % 5 == 0:
                progress = ((i + 1) / len(positive_thresholds)) * 100
                self.logger.info(f"Progress: {progress:.1f}%")
        
        # Generate optimization report
        report_path = self._generate_optimization_report(
            results,
            best_thresholds,
            best_score,
            metric,
            validation_data,
            output_dir,
        )
        
        optimization_result = {
            "best_thresholds": {
                "positive": float(best_thresholds[0]),
                "negative": float(best_thresholds[1]),
            },
            "best_score": float(best_score),
            "metric_optimized": metric,
            "total_combinations_tested": len(results),
            "all_results": results,
            "report_path": report_path,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save results to JSON
        results_file = Path(output_dir) / "threshold_optimization_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(optimization_result, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            f"Optimization completed. Best thresholds: "
            f"positive={best_thresholds[0]:.3f}, negative={best_thresholds[1]:.3f}, "
            f"{metric}={best_score:.4f}"
        )
        
        return optimization_result

    def _load_validation_data(self, validation_file: str) -> List[Dict[str, Any]]:
        """Load validation dataset."""
        validation_path = Path(validation_file)
        
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation file not found: {validation_file}")
        
        with open(validation_path, "r", encoding="utf-8") as f:
            validation_data = json.load(f)
        
        self.logger.info(f"Loaded {len(validation_data)} validation examples")
        return validation_data

    def _evaluate_thresholds(
        self,
        validation_data: List[Dict],
        positive_threshold: float,
        negative_threshold: float,
        metric: str,
        output_dir: str,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a specific threshold combination.
        
        Args:
            validation_data: Validation dataset
            positive_threshold: Positive threshold to test
            negative_threshold: Negative threshold to test
            metric: Metric to calculate
            output_dir: Output directory
            
        Returns:
            Tuple of (metric_score, all_metrics_dict)
        """
        # Update engine thresholds
        self.engine.sentiment_analyzer.update_thresholds(
            positive_threshold, negative_threshold
        )
        
        # Prepare temporary input file
        temp_file = Path(output_dir) / "temp_optimization_input.json"
        input_data = [
            {
                "review_id": item.get("review_id", f"OPT{i:03d}"),
                "review_text": item["review_text"],
            }
            for i, item in enumerate(validation_data)
        ]
        
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f, indent=2, ensure_ascii=False)
        
        # Run analysis
        try:
            results = self.engine.analyze_reviews(str(temp_file), output_dir)
            
            # Extract predictions
            import csv
            results_file = results["files_generated"]["results"]
            predictions_dict = {}
            
            with open(results_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    predictions_dict[row["review_id"]] = row["sentiment_final"]
            
            # Match with expected
            predictions = []
            expected = []
            
            for item in validation_data:
                review_id = item.get("review_id", "")
                if review_id in predictions_dict:
                    predictions.append(predictions_dict[review_id])
                    expected.append(item["expected_sentiment"])
            
            # Calculate metrics
            accuracy = accuracy_score(expected, predictions)
            f1 = f1_score(expected, predictions, average="weighted", zero_division=0)
            balanced_acc = balanced_accuracy_score(expected, predictions)
            
            metrics_dict = {
                "accuracy": accuracy,
                "f1_score": f1,
                "balanced_accuracy": balanced_acc,
            }
            
            # Return requested metric
            metric_value = metrics_dict.get(metric, f1)
            
            return metric_value, metrics_dict
            
        except Exception as e:
            self.logger.error(f"Error evaluating thresholds: {e}")
            return 0.0, {"accuracy": 0.0, "f1_score": 0.0, "balanced_accuracy": 0.0}
        finally:
            # Clean up
            temp_file.unlink(missing_ok=True)

    def _generate_optimization_report(
        self,
        results: List[Dict],
        best_thresholds: Tuple[float, float],
        best_score: float,
        metric: str,
        validation_data: List[Dict],
        output_dir: str,
    ) -> str:
        """Generate optimization report with visualizations."""
        report_path = Path(output_dir) / "THRESHOLD_OPTIMIZATION_REPORT.md"
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Generate heatmap visualization
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create pivot table for heatmap
            pivot_data = df.pivot_table(
                values="score",
                index="positive_threshold",
                columns="negative_threshold",
                aggfunc="mean",
            )
            
            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(
                pivot_data,
                cmap="RdYlGn",
                annot=False,
                fmt=".3f",
                cbar_kws={"label": metric},
            )
            plt.title(
                f"Optimisation des Seuils - {metric.upper()}\n"
                f"Meilleur: pos={best_thresholds[0]:.3f}, neg={best_thresholds[1]:.3f}, "
                f"score={best_score:.4f}",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Seuil Négatif", fontsize=12)
            plt.ylabel("Seuil Positif", fontsize=12)
            plt.tight_layout()
            
            heatmap_path = Path(output_dir) / "threshold_optimization_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            has_visualization = True
            
        except ImportError:
            self.logger.warning(
                "matplotlib/seaborn not installed. Skipping visualization."
            )
            has_visualization = False
        except Exception as e:
            self.logger.error(f"Error generating visualization: {e}")
            has_visualization = False
        
        # Generate report content
        report_lines = [
            "# Rapport d'Optimisation des Seuils",
            "",
            f"**Date** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Dataset de validation** : {len(validation_data)} exemples",
            f"**Métrique optimisée** : {metric}",
            "",
            "## 🎯 Résultats Optimaux",
            "",
            f"- **Seuil Positif Optimal** : {best_thresholds[0]:.4f}",
            f"- **Seuil Négatif Optimal** : {best_thresholds[1]:.4f}",
            f"- **Score Optimal ({metric})** : {best_score:.4f}",
            f"- **Zone Neutre** : [{best_thresholds[1]:.4f}, {best_thresholds[0]:.4f}]",
            "",
            "## 📊 Statistiques de Recherche",
            "",
            f"- **Combinaisons testées** : {len(results)}",
            f"- **Score minimum** : {df['score'].min():.4f}",
            f"- **Score maximum** : {df['score'].max():.4f}",
            f"- **Score moyen** : {df['score'].mean():.4f}",
            f"- **Écart-type** : {df['score'].std():.4f}",
            "",
        ]
        
        if has_visualization:
            report_lines.extend(
                [
                    "## 📈 Visualisation",
                    "",
                    "![Heatmap d'optimisation](threshold_optimization_heatmap.png)",
                    "",
                ]
            )
        
        # Top 10 configurations
        top_10 = df.nlargest(10, "score")
        report_lines.extend(
            [
                "## 🏆 Top 10 des Configurations",
                "",
                "| Rang | Seuil Positif | Seuil Négatif | Score | Accuracy | F1-Score |",
                "|------|---------------|---------------|-------|----------|----------|",
            ]
        )
        
        for i, row in enumerate(top_10.itertuples(), 1):
            report_lines.append(
                f"| {i} | {row.positive_threshold:.4f} | {row.negative_threshold:.4f} | "
                f"{row.score:.4f} | {row.accuracy:.4f} | {row.f1_score:.4f} |"
            )
        
        report_lines.extend(
            [
                "",
                "## 💡 Analyse et Recommandations",
                "",
                self._generate_threshold_analysis(best_thresholds, df),
                "",
                "## 🔧 Application des Seuils Optimaux",
                "",
                "Pour appliquer ces seuils optimaux, mettez à jour votre `config.json` :",
                "",
                "```json",
                "{",
                '  "sentiment_thresholds": {',
                f'    "positive": {best_thresholds[0]:.4f},',
                f'    "negative": {best_thresholds[1]:.4f}',
                "  }",
                "}",
                "```",
                "",
                "## 📝 Justification Scientifique",
                "",
                f"Les seuils optimaux ont été déterminés par recherche exhaustive sur {len(results)} "
                f"combinaisons de paramètres, en maximisant le {metric} sur un dataset de validation "
                f"de {len(validation_data)} exemples annotés manuellement. Cette approche garantit une "
                "classification optimale basée sur des données empiriques plutôt que sur des choix arbitraires.",
            ]
        )
        
        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f"Optimization report generated: {report_path}")
        return str(report_path)

    def _generate_threshold_analysis(
        self, best_thresholds: Tuple[float, float], df: pd.DataFrame
    ) -> str:
        """Generate analysis of optimal thresholds."""
        pos_thresh, neg_thresh = best_thresholds
        neutral_zone = pos_thresh - neg_thresh
        
        analysis = [
            f"### Zone Neutre : {neutral_zone:.4f}",
            "",
            f"La zone neutre s'étend de {neg_thresh:.4f} à {pos_thresh:.4f}, "
            f"soit une largeur de {neutral_zone:.4f}.",
            "",
        ]
        
        if neutral_zone < 0.05:
            analysis.append(
                "⚠️ **Zone neutre étroite** : Peu de sentiments seront classés comme neutres. "
                "Cela peut être approprié si votre dataset contient peu de sentiments ambigus."
            )
        elif neutral_zone > 0.15:
            analysis.append(
                "⚠️ **Zone neutre large** : Beaucoup de sentiments seront classés comme neutres. "
                "Cela favorise la prudence mais peut réduire la sensibilité de la classification."
            )
        else:
            analysis.append(
                "✅ **Zone neutre équilibrée** : La largeur de la zone neutre est appropriée "
                "pour une classification équilibrée entre les trois catégories."
            )
        
        analysis.extend(
            [
                "",
                "### Sensibilité",
                "",
                f"- **Seuil positif** : {pos_thresh:.4f} - "
                f"{'Élevé (classification stricte)' if pos_thresh > 0.1 else 'Bas (classification permissive)'}",
                f"- **Seuil négatif** : {neg_thresh:.4f} - "
                f"{'Élevé (classification stricte)' if abs(neg_thresh) > 0.1 else 'Bas (classification permissive)'}",
            ]
        )
        
        return "\n".join(analysis)

    def compare_analyzers(
        self,
        validation_file: str,
        analyzers: List[Tuple[str, Any]],
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        """
        Compare different analyzer types with optimal thresholds.
        
        Args:
            validation_file: Path to validation dataset
            analyzers: List of (name, analyzer_instance) tuples
            output_dir: Directory to save comparison results
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info(f"Comparing {len(analyzers)} analyzer types")
        
        results = []
        
        for name, analyzer in analyzers:
            self.logger.info(f"Optimizing thresholds for {name}")
            
            # Create temporary engine with this analyzer
            temp_engine = SentimentAnalysisEngine()
            temp_engine.sentiment_analyzer = analyzer
            
            # Create optimizer for this engine
            optimizer = ThresholdOptimizer(temp_engine)
            
            # Find optimal thresholds
            optimization_result = optimizer.find_optimal_thresholds(
                validation_file, output_dir=output_dir
            )
            
            results.append(
                {
                    "analyzer_name": name,
                    "best_thresholds": optimization_result["best_thresholds"],
                    "best_score": optimization_result["best_score"],
                    "analyzer_info": analyzer.get_analyzer_info(),
                }
            )
        
        # Find best analyzer
        best_analyzer = max(results, key=lambda x: x["best_score"])
        
        comparison_report = {
            "analyzers_compared": len(analyzers),
            "results": results,
            "best_analyzer": best_analyzer,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save comparison
        comparison_file = Path(output_dir) / "analyzer_comparison.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            f"Best analyzer: {best_analyzer['analyzer_name']} "
            f"(score: {best_analyzer['best_score']:.4f})"
        )
        
        return comparison_report
