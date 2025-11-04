"""Sentiment analysis validation and quality metrics."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from ..models.review import Review
from ..engine import SentimentAnalysisEngine


class SentimentValidator:
    """
    Validates sentiment analysis quality using annotated datasets.
    
    Calculates comprehensive metrics including accuracy, precision, recall,
    F1-score, and generates detailed validation reports.
    """

    def __init__(self, engine: SentimentAnalysisEngine = None):
        """
        Initialize validator.
        
        Args:
            engine: SentimentAnalysisEngine instance to validate
        """
        self.engine = engine or SentimentAnalysisEngine()
        self.logger = logging.getLogger(__name__)

    def load_validation_data(self, validation_file: str) -> List[Dict[str, Any]]:
        """
        Load validation dataset with expected sentiments.
        
        Args:
            validation_file: Path to JSON file with annotated reviews
            
        Returns:
            List of validation data dictionaries
            
        Raises:
            FileNotFoundError: If validation file doesn't exist
            ValueError: If validation data is invalid
        """
        validation_path = Path(validation_file)
        
        if not validation_path.exists():
            raise FileNotFoundError(f"Validation file not found: {validation_file}")
        
        self.logger.info(f"Loading validation data from {validation_file}")
        
        with open(validation_path, "r", encoding="utf-8") as f:
            validation_data = json.load(f)
        
        # Validate structure
        if not isinstance(validation_data, list):
            raise ValueError("Validation data must be a list of dictionaries")
        
        for i, item in enumerate(validation_data):
            if "review_text" not in item:
                raise ValueError(f"Item {i} missing 'review_text' field")
            if "expected_sentiment" not in item:
                raise ValueError(f"Item {i} missing 'expected_sentiment' field")
            if item["expected_sentiment"] not in ["Positive", "Negative", "Neutral"]:
                raise ValueError(
                    f"Item {i} has invalid expected_sentiment: {item['expected_sentiment']}"
                )
        
        self.logger.info(f"Loaded {len(validation_data)} validation examples")
        return validation_data

    def validate_analyzer(
        self, validation_file: str, output_dir: str = "."
    ) -> Dict[str, Any]:
        """
        Validate sentiment analyzer and calculate quality metrics.
        
        Args:
            validation_file: Path to validation dataset
            output_dir: Directory to save validation results
            
        Returns:
            Dictionary with validation metrics and results
        """
        self.logger.info("Starting sentiment analyzer validation")
        
        # Load validation data
        validation_data = self.load_validation_data(validation_file)
        
        # Prepare temporary file for analysis
        temp_file = Path(output_dir) / "temp_validation_input.json"
        self._prepare_validation_input(validation_data, temp_file)
        
        # Run analysis
        self.logger.info("Running sentiment analysis on validation data")
        results = self.engine.analyze_reviews(str(temp_file), output_dir)
        
        # Extract predictions and expected values
        predictions, expected = self._extract_predictions_and_expected(
            validation_data, results
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(expected, predictions)
        
        # Generate detailed report
        report_path = self._generate_validation_report(
            metrics, expected, predictions, validation_data, output_dir
        )
        
        # Clean up temporary file
        temp_file.unlink(missing_ok=True)
        
        self.logger.info(
            f"Validation completed - Accuracy: {metrics['accuracy']:.2%}, "
            f"F1-Score: {metrics['f1_score']:.2%}"
        )
        
        return {
            "metrics": metrics,
            "report_path": report_path,
            "validation_size": len(validation_data),
            "timestamp": datetime.now().isoformat(),
        }

    def _prepare_validation_input(
        self, validation_data: List[Dict], output_file: Path
    ) -> None:
        """Prepare validation data in format expected by engine."""
        input_data = [
            {
                "review_id": item.get("review_id", f"VAL{i:03d}"),
                "review_text": item["review_text"],
            }
            for i, item in enumerate(validation_data)
        ]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(input_data, f, indent=2, ensure_ascii=False)

    def _extract_predictions_and_expected(
        self, validation_data: List[Dict], results: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Extract predictions and expected values from results."""
        # Load detailed results from CSV
        import csv
        
        results_file = results["files_generated"]["results"]
        predictions_dict = {}
        
        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                predictions_dict[row["review_id"]] = row["sentiment_final"]
        
        # Match predictions with expected values
        predictions = []
        expected = []
        
        for item in validation_data:
            review_id = item.get("review_id", "")
            if review_id in predictions_dict:
                predictions.append(predictions_dict[review_id])
                expected.append(item["expected_sentiment"])
        
        return predictions, expected

    def _calculate_metrics(
        self, expected: List[str], predictions: List[str]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive quality metrics.
        
        Args:
            expected: List of expected sentiment labels
            predictions: List of predicted sentiment labels
            
        Returns:
            Dictionary with all metrics
        """
        # Overall metrics
        accuracy = accuracy_score(expected, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            expected, predictions, average="weighted", zero_division=0
        )
        
        # Per-class detailed metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(
                expected, predictions, average=None, zero_division=0,
                labels=["Positive", "Negative", "Neutral"]
            )
        )
        
        # Confusion matrix
        cm = confusion_matrix(
            expected, predictions, labels=["Positive", "Negative", "Neutral"]
        )
        
        # Classification report
        class_report = classification_report(
            expected,
            predictions,
            labels=["Positive", "Negative", "Neutral"],
            output_dict=True,
            zero_division=0,
        )
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "per_class_metrics": {
                "Positive": {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1_score": float(f1_per_class[0]),
                    "support": int(support_per_class[0]),
                },
                "Negative": {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1_score": float(f1_per_class[1]),
                    "support": int(support_per_class[1]),
                },
                "Neutral": {
                    "precision": float(precision_per_class[2]),
                    "recall": float(recall_per_class[2]),
                    "f1_score": float(f1_per_class[2]),
                    "support": int(support_per_class[2]),
                },
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
        }
        
        return metrics

    def _generate_validation_report(
        self,
        metrics: Dict[str, Any],
        expected: List[str],
        predictions: List[str],
        validation_data: List[Dict],
        output_dir: str,
    ) -> str:
        """Generate detailed validation report."""
        report_path = Path(output_dir) / "VALIDATION_REPORT.md"
        
        # Find misclassifications
        misclassifications = []
        for i, (exp, pred) in enumerate(zip(expected, predictions)):
            if exp != pred:
                misclassifications.append(
                    {
                        "review_id": validation_data[i].get("review_id", f"VAL{i:03d}"),
                        "text": validation_data[i]["review_text"],
                        "expected": exp,
                        "predicted": pred,
                    }
                )
        
        # Generate report content
        report_lines = [
            "# Rapport de Validation - Analyse de Sentiment",
            "",
            f"**Date de validation** : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Taille du dataset** : {len(validation_data)} exemples",
            "",
            "## 📊 Métriques Globales",
            "",
            f"- **Accuracy (Précision globale)** : {metrics['accuracy']:.2%}",
            f"- **Precision (Précision pondérée)** : {metrics['precision']:.2%}",
            f"- **Recall (Rappel pondéré)** : {metrics['recall']:.2%}",
            f"- **F1-Score (Score F1 pondéré)** : {metrics['f1_score']:.2%}",
            "",
            "## 📈 Métriques par Classe",
            "",
            "### Sentiment Positif",
            f"- Precision : {metrics['per_class_metrics']['Positive']['precision']:.2%}",
            f"- Recall : {metrics['per_class_metrics']['Positive']['recall']:.2%}",
            f"- F1-Score : {metrics['per_class_metrics']['Positive']['f1_score']:.2%}",
            f"- Support : {metrics['per_class_metrics']['Positive']['support']} exemples",
            "",
            "### Sentiment Négatif",
            f"- Precision : {metrics['per_class_metrics']['Negative']['precision']:.2%}",
            f"- Recall : {metrics['per_class_metrics']['Negative']['recall']:.2%}",
            f"- F1-Score : {metrics['per_class_metrics']['Negative']['f1_score']:.2%}",
            f"- Support : {metrics['per_class_metrics']['Negative']['support']} exemples",
            "",
            "### Sentiment Neutre",
            f"- Precision : {metrics['per_class_metrics']['Neutral']['precision']:.2%}",
            f"- Recall : {metrics['per_class_metrics']['Neutral']['recall']:.2%}",
            f"- F1-Score : {metrics['per_class_metrics']['Neutral']['f1_score']:.2%}",
            f"- Support : {metrics['per_class_metrics']['Neutral']['support']} exemples",
            "",
            "## 🔀 Matrice de Confusion",
            "",
            "```",
            "                 Prédit Positif  Prédit Négatif  Prédit Neutre",
            f"Réel Positif     {metrics['confusion_matrix'][0][0]:14d}  {metrics['confusion_matrix'][0][1]:14d}  {metrics['confusion_matrix'][0][2]:13d}",
            f"Réel Négatif     {metrics['confusion_matrix'][1][0]:14d}  {metrics['confusion_matrix'][1][1]:14d}  {metrics['confusion_matrix'][1][2]:13d}",
            f"Réel Neutre      {metrics['confusion_matrix'][2][0]:14d}  {metrics['confusion_matrix'][2][1]:14d}  {metrics['confusion_matrix'][2][2]:13d}",
            "```",
            "",
            "## ❌ Erreurs de Classification",
            "",
            f"**Nombre total d'erreurs** : {len(misclassifications)} / {len(validation_data)} ({len(misclassifications)/len(validation_data)*100:.1f}%)",
            "",
        ]
        
        if misclassifications:
            report_lines.append("### Détail des Erreurs")
            report_lines.append("")
            for i, error in enumerate(misclassifications[:20], 1):  # Limit to 20
                report_lines.extend(
                    [
                        f"#### Erreur {i} - {error['review_id']}",
                        f"- **Texte** : \"{error['text'][:100]}{'...' if len(error['text']) > 100 else ''}\"",
                        f"- **Attendu** : {error['expected']}",
                        f"- **Prédit** : {error['predicted']}",
                        "",
                    ]
                )
            
            if len(misclassifications) > 20:
                report_lines.append(
                    f"*... et {len(misclassifications) - 20} autres erreurs*"
                )
                report_lines.append("")
        
        # Add interpretation
        report_lines.extend(
            [
                "## 💡 Interprétation des Résultats",
                "",
                self._interpret_results(metrics),
                "",
                "## 🎯 Recommandations",
                "",
                self._generate_recommendations(metrics, misclassifications),
            ]
        )
        
        # Write report
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        
        self.logger.info(f"Validation report generated: {report_path}")
        return str(report_path)

    def _interpret_results(self, metrics: Dict[str, Any]) -> str:
        """Generate interpretation of validation results."""
        accuracy = metrics["accuracy"]
        
        if accuracy >= 0.90:
            quality = "**Excellente**"
            comment = "Le modèle atteint une précision exceptionnelle."
        elif accuracy >= 0.80:
            quality = "**Très bonne**"
            comment = "Le modèle montre une bonne performance globale."
        elif accuracy >= 0.70:
            quality = "**Bonne**"
            comment = "Le modèle est fonctionnel mais peut être amélioré."
        elif accuracy >= 0.60:
            quality = "**Acceptable**"
            comment = "Le modèle nécessite des améliorations significatives."
        else:
            quality = "**Insuffisante**"
            comment = "Le modèle nécessite une révision complète."
        
        return f"La qualité globale du modèle est {quality} avec une accuracy de {accuracy:.2%}. {comment}"

    def _generate_recommendations(
        self, metrics: Dict[str, Any], misclassifications: List[Dict]
    ) -> str:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check per-class performance
        for sentiment in ["Positive", "Negative", "Neutral"]:
            class_metrics = metrics["per_class_metrics"][sentiment]
            if class_metrics["f1_score"] < 0.70:
                recommendations.append(
                    f"- Améliorer la détection des sentiments **{sentiment}** "
                    f"(F1-Score actuel: {class_metrics['f1_score']:.2%})"
                )
        
        # Check for common error patterns
        if misclassifications:
            # Analyze error patterns
            neutral_confused_with_positive = sum(
                1
                for e in misclassifications
                if e["expected"] == "Neutral" and e["predicted"] == "Positive"
            )
            neutral_confused_with_negative = sum(
                1
                for e in misclassifications
                if e["expected"] == "Neutral" and e["predicted"] == "Negative"
            )
            
            if neutral_confused_with_positive > len(misclassifications) * 0.3:
                recommendations.append(
                    "- Adjust thresholds: too many neutral sentiments classified as positive"
                )
            if neutral_confused_with_negative > len(misclassifications) * 0.3:
                recommendations.append(
                    "- Adjust thresholds: too many neutral sentiments classified as negative"
                )
        
        # General recommendations
        if metrics["accuracy"] < 0.85:
            recommendations.append(
                "- Consider using a pre-trained ML model for French (CamemBERT, FlauBERT)"
            )
            recommendations.append(
                "- Extend the French word dictionary with domain-specific terms"
            )
        
        if not recommendations:
            recommendations.append(
                "- Model performs well. Continue monitoring on new data."
            )
        
        return "\n".join(recommendations)

    def compare_configurations(
        self,
        validation_file: str,
        configurations: List[Dict[str, Any]],
        output_dir: str = ".",
    ) -> Dict[str, Any]:
        """
        Compare multiple analyzer configurations.
        
        Args:
            validation_file: Path to validation dataset
            configurations: List of configuration dictionaries to test
            output_dir: Directory to save comparison results
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info(f"Comparing {len(configurations)} configurations")
        
        results = []
        
        for i, config in enumerate(configurations):
            self.logger.info(f"Testing configuration {i+1}/{len(configurations)}")
            
            # Update engine configuration
            self.engine.update_configuration(config)
            
            # Validate
            validation_result = self.validate_analyzer(validation_file, output_dir)
            
            results.append(
                {
                    "configuration": config,
                    "metrics": validation_result["metrics"],
                }
            )
        
        # Find best configuration
        best_config = max(results, key=lambda x: x["metrics"]["f1_score"])
        
        comparison_report = {
            "configurations_tested": len(configurations),
            "results": results,
            "best_configuration": best_config,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save comparison report
        comparison_file = Path(output_dir) / "configuration_comparison.json"
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(
            f"Best configuration achieved F1-Score: {best_config['metrics']['f1_score']:.2%}"
        )
        
        return comparison_report
