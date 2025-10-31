#!/usr/bin/env python3
"""
Script de validation de l'installation du moteur d'analyse de sentiment.

Ce script vérifie que tous les composants sont correctement installés
et fonctionnent comme attendu.
"""

import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path


def print_header(title):
    """Affiche un en-tête formaté."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def print_step(step_name, status="RUNNING"):
    """Affiche l'état d'une étape."""
    status_symbols = {
        "RUNNING": "⏳",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️"
    }
    symbol = status_symbols.get(status, "❓")
    print(f"{symbol} {step_name}")


def check_python_version():
    """Vérifie la version de Python."""
    print_step("Vérification de la version Python", "RUNNING")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_step(f"Python {version.major}.{version.minor}.{version.micro} - OK", "SUCCESS")
        return True
    else:
        print_step(f"Python {version.major}.{version.minor}.{version.micro} - Version insuffisante (requis: 3.8+)", "ERROR")
        return False


def check_dependencies():
    """Vérifie les dépendances Python."""
    print_step("Vérification des dépendances", "RUNNING")
    
    required_packages = [
        "vaderSentiment",
        "pandas",
        "numpy",
        "chardet",
        "psutil"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_step(f"  {package} - OK", "SUCCESS")
        except ImportError:
            print_step(f"  {package} - MANQUANT", "ERROR")
            missing_packages.append(package)
    
    if missing_packages:
        print_step(f"Dépendances manquantes: {', '.join(missing_packages)}", "ERROR")
        return False
    else:
        print_step("Toutes les dépendances sont installées", "SUCCESS")
        return True


def check_project_structure():
    """Vérifie la structure du projet."""
    print_step("Vérification de la structure du projet", "RUNNING")
    
    required_files = [
        "main.py",
        "requirements.txt",
        "config.json",
        "README.md",
        "src/__init__.py",
        "src/models/review.py",
        "src/data/loader.py",
        "src/preprocessing/preprocessor.py",
        "src/analysis/sentiment_analyzer.py",
        "src/reporting/report_generator.py",
        "src/config/config_manager.py",
        "src/engine.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print_step(f"  {file_path} - OK", "SUCCESS")
        else:
            print_step(f"  {file_path} - MANQUANT", "ERROR")
            missing_files.append(file_path)
    
    if missing_files:
        print_step(f"Fichiers manquants: {', '.join(missing_files)}", "ERROR")
        return False
    else:
        print_step("Structure du projet correcte", "SUCCESS")
        return True


def test_basic_functionality():
    """Teste les fonctionnalités de base."""
    print_step("Test des fonctionnalités de base", "RUNNING")
    
    try:
        # Import des modules principaux
        from src.models.review import Review, SentimentResult
        from src.data.loader import JSONDataLoader
        from src.analysis.sentiment_analyzer import VaderSentimentAnalyzer
        from src.engine import SentimentAnalysisEngine
        
        print_step("  Import des modules - OK", "SUCCESS")
        
        # Test de création d'objets
        review = Review(review_id="TEST001", review_text="Test review")
        loader = JSONDataLoader()
        analyzer = VaderSentimentAnalyzer()
        engine = SentimentAnalysisEngine()
        
        print_step("  Création d'objets - OK", "SUCCESS")
        
        # Test d'analyse simple
        test_review = Review(review_id="TEST002", review_text="Excellent produit !")
        analyzed = analyzer.analyze_single_review(test_review)
        
        if analyzed.sentiment_label in ["Positive", "Negative", "Neutral"]:
            print_step("  Analyse de sentiment - OK", "SUCCESS")
        else:
            print_step("  Analyse de sentiment - ERREUR", "ERROR")
            return False
        
        return True
        
    except Exception as e:
        print_step(f"Erreur lors du test: {e}", "ERROR")
        return False


def test_end_to_end():
    """Test du pipeline complet."""
    print_step("Test du pipeline complet", "RUNNING")
    
    try:
        # Créer des données de test
        test_data = [
            {"review_id": "VAL001", "review_text": "Excellent produit, je le recommande !"},
            {"review_id": "VAL002", "review_text": "Service client horrible."},
            {"review_id": "VAL003", "review_text": "Produit correct, sans plus."},
            {"review_id": "VAL004", "review_text": ""},  # Test avec texte vide
            {"review_id": "VAL005", "review_text": "Très satisfait ! 😀"}
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Créer le fichier d'entrée
            input_file = os.path.join(temp_dir, "test_reviews.json")
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            
            print_step("  Fichier de test créé - OK", "SUCCESS")
            
            # Exécuter l'analyse
            from src.engine import SentimentAnalysisEngine
            engine = SentimentAnalysisEngine()
            
            results = engine.analyze_reviews(input_file, temp_dir)
            
            # Vérifier les résultats
            if results["status"] == "completed":
                print_step("  Pipeline exécuté - OK", "SUCCESS")
            else:
                print_step("  Pipeline exécuté - ERREUR", "ERROR")
                return False
            
            if results["statistics"]["total_reviews"] == 5:
                print_step("  Nombre de reviews - OK", "SUCCESS")
            else:
                print_step("  Nombre de reviews - ERREUR", "ERROR")
                return False
            
            # Vérifier les fichiers de sortie
            summary_file = os.path.join(temp_dir, "summary.json")
            results_file = os.path.join(temp_dir, "results.csv")
            
            if os.path.exists(summary_file) and os.path.exists(results_file):
                print_step("  Fichiers de sortie générés - OK", "SUCCESS")
            else:
                print_step("  Fichiers de sortie générés - ERREUR", "ERROR")
                return False
            
            return True
            
    except Exception as e:
        print_step(f"Erreur lors du test E2E: {e}", "ERROR")
        return False


def test_command_line():
    """Test de l'interface en ligne de commande."""
    print_step("Test de l'interface en ligne de commande", "RUNNING")
    
    try:
        # Test de l'aide
        result = subprocess.run([sys.executable, "main.py", "--help"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and "Sentiment Analysis Engine" in result.stdout:
            print_step("  Commande --help - OK", "SUCCESS")
        else:
            print_step("  Commande --help - ERREUR", "ERROR")
            return False
        
        # Test de validation
        if os.path.exists("reviews.json"):
            result = subprocess.run([sys.executable, "main.py", "--validate-only"], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print_step("  Validation du fichier - OK", "SUCCESS")
            else:
                print_step("  Validation du fichier - ERREUR", "ERROR")
                return False
        else:
            print_step("  Fichier reviews.json non trouvé - IGNORÉ", "WARNING")
        
        return True
        
    except Exception as e:
        print_step(f"Erreur lors du test CLI: {e}", "ERROR")
        return False


def run_unit_tests():
    """Exécute les tests unitaires."""
    print_step("Exécution des tests unitaires", "RUNNING")
    
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Compter les tests passés
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if "passed" in line:
                    print_step(f"  Tests unitaires - {line.strip()}", "SUCCESS")
                    break
            return True
        else:
            print_step(f"  Tests unitaires - ÉCHEC", "ERROR")
            print(f"Sortie d'erreur: {result.stderr}")
            return False
            
    except Exception as e:
        print_step(f"Erreur lors des tests unitaires: {e}", "ERROR")
        return False


def main():
    """Fonction principale de validation."""
    print_header("VALIDATION DE L'INSTALLATION")
    print("Ce script vérifie que le moteur d'analyse de sentiment")
    print("est correctement installé et fonctionnel.")
    
    # Liste des vérifications
    checks = [
        ("Version Python", check_python_version),
        ("Dépendances", check_dependencies),
        ("Structure du projet", check_project_structure),
        ("Fonctionnalités de base", test_basic_functionality),
        ("Pipeline complet", test_end_to_end),
        ("Interface en ligne de commande", test_command_line),
        ("Tests unitaires", run_unit_tests)
    ]
    
    results = {}
    
    for check_name, check_function in checks:
        print_header(f"VÉRIFICATION: {check_name}")
        results[check_name] = check_function()
    
    # Résumé final
    print_header("RÉSUMÉ DE LA VALIDATION")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for check_name, success in results.items():
        status = "SUCCESS" if success else "ERROR"
        print_step(f"{check_name}", status)
    
    print(f"\nRésultat: {success_count}/{total_count} vérifications réussies")
    
    if success_count == total_count:
        print_step("INSTALLATION VALIDÉE AVEC SUCCÈS", "SUCCESS")
        print("\n🎉 Le moteur d'analyse de sentiment est prêt à être utilisé !")
        print("\nCommandes de base:")
        print("  python main.py reviews.json")
        print("  python main.py --help")
        return 0
    else:
        print_step("PROBLÈMES DÉTECTÉS DANS L'INSTALLATION", "ERROR")
        print("\n❌ Veuillez corriger les erreurs avant d'utiliser le système.")
        return 1


if __name__ == "__main__":
    sys.exit(main())