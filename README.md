# Moteur d'Analyse de Sentiment

Une solution Python complète et robuste pour analyser le sentiment d'avis clients avec génération de rapports détaillés.

## 🎯 Objectif

Ce moteur d'analyse de sentiment traite des corpus d'avis clients en français, les classe automatiquement (Positif, Négatif, Neutre) et génère des rapports synthétiques pour l'aide à la décision.

## ✨ Fonctionnalités

- **Analyse de sentiment robuste** : Classification automatique avec VADER optimisé pour le français
- **Gestion des cas complexes** : Emojis, caractères spéciaux, textes vides, encodages variés
- **Rapports complets** : Statistiques détaillées et export CSV
- **Interface en ligne de commande** : Utilisation simple et intuitive
- **Architecture modulaire** : Code maintenable et extensible
- **Gestion d'erreurs avancée** : Le système ne plante jamais

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation rapide

```bash
# Cloner le projet
git clone <repository-url>
cd UserFeedbackAnalyzer-Python

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Linux/macOS :
source venv/bin/activate
# Sur Windows :
# venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 📖 Utilisation

### Utilisation basique

```bash
# Analyser le fichier reviews.json par défaut
python main.py

# Analyser un fichier spécifique
python main.py mon_fichier.json

# Spécifier un répertoire de sortie
python main.py reviews.json --output-dir ./resultats
```

### Options avancées

```bash
# Utiliser une configuration personnalisée
python main.py reviews.json --config ma_config.json

# Mode verbeux pour plus de détails
python main.py reviews.json --verbose

# Mode silencieux (erreurs uniquement)
python main.py reviews.json --quiet

# Valider uniquement le fichier d'entrée
python main.py reviews.json --validate-only

# Afficher l'aide
python main.py --help
```

## 📁 Format des données d'entrée

Le fichier d'entrée doit être au format JSON avec la structure suivante :

```json
[
  {
    "review_id": "REV001",
    "review_text": "Excellent produit, je le recommande vivement !"
  },
  {
    "review_id": "REV002", 
    "review_text": "Service client décevant."
  }
]
```

### Formats supportés

- **Structure simple** : Liste d'objets avec `review_id` et `review_text`
- **Structure encapsulée** : `{"reviews": [...]}`
- **Champs alternatifs** : `id`, `text`, `content` sont automatiquement détectés
- **Gestion robuste** : Textes vides, caractères spéciaux, emojis

## 📊 Fichiers de sortie

### Rapport de synthèse (`summary.json`)

```json
{
  "analysis_summary": {
    "total_reviews": 50,
    "sentiment_distribution": {
      "positive": {"count": 12, "percentage": 24.0},
      "negative": {"count": 9, "percentage": 18.0},
      "neutral": {"count": 29, "percentage": 58.0}
    },
    "processing_info": {
      "timestamp": "2025-10-31T10:00:00Z",
      "errors_count": 2,
      "configuration": {
        "positive_threshold": 0.05,
        "negative_threshold": -0.05
      }
    }
  }
}
```

### Résultats détaillés (`results.csv`)

```csv
review_id,review_text,sentiment_final,sentiment_score,processing_errors
REV001,"Excellent produit !",Positive,0.8516,
REV002,"Service décevant",Negative,-0.7269,
REV003,"",Neutral,0.0,Review text is empty
```

## ⚙️ Configuration

### Fichier de configuration (`config.json`)

```json
{
  "sentiment_thresholds": {
    "positive": 0.05,
    "negative": -0.05
  },
  "output": {
    "summary_format": "json",
    "results_format": "csv"
  },
  "logging": {
    "level": "INFO",
    "file": "sentiment_analysis.log"
  }
}
```

### Justification des seuils

- **Seuil positif (0.05)** : Score VADER > 0.05 pour classification positive
- **Seuil négatif (-0.05)** : Score VADER < -0.05 pour classification négative  
- **Zone neutre** : Entre -0.05 et 0.05 pour les sentiments ambigus

Ces seuils créent une classification équilibrée avec une zone neutre pour les sentiments ambigus, optimisée pour les avis clients en français.

## 🏗️ Architecture

### Structure du projet

```
sentiment_analysis_engine/
├── src/                    # Code source principal
│   ├── models/            # Modèles de données
│   ├── data/              # Chargement des données
│   ├── preprocessing/     # Prétraitement du texte
│   ├── analysis/          # Analyse de sentiment
│   ├── reporting/         # Génération de rapports
│   ├── config/            # Gestion de configuration
│   └── utils/             # Utilitaires
├── tests/                 # Tests unitaires
├── main.py               # Point d'entrée
├── config.json           # Configuration par défaut
└── requirements.txt      # Dépendances
```

### Composants principaux

1. **DataLoader** : Chargement robuste des fichiers JSON
2. **ReviewPreprocessor** : Nettoyage et normalisation du texte
3. **VaderSentimentAnalyzer** : Analyse de sentiment avec optimisations françaises
4. **ReportGenerator** : Génération des rapports de synthèse et détaillés
5. **SentimentAnalysisEngine** : Orchestrateur principal du pipeline

## 🧪 Tests

### Exécuter les tests

```bash
# Tests unitaires
python -m pytest tests/ -v

# Tests avec couverture de code
python -m pytest tests/ --cov=src --cov-report=term-missing

# Tests d'intégration uniquement
python -m pytest tests/test_integration.py -v
```

### Couverture de code

Le projet maintient une couverture de code > 60% avec des tests complets pour :
- Tous les composants principaux
- Cas d'erreur et cas limites
- Pipeline d'intégration complet
- Gestion des données problématiques

## 🔧 Développement

### Qualité du code

```bash
# Formatage automatique avec Black
black src/ tests/

# Vérification du style avec Flake8
flake8 src/ tests/

# Exécution complète des vérifications
black src/ tests/ && flake8 src/ tests/ && python -m pytest tests/
```

### Ajout de nouvelles fonctionnalités

1. **Nouveaux analyseurs** : Implémenter `SentimentAnalyzerInterface`
2. **Nouveaux formats** : Étendre `DataLoaderInterface`
3. **Nouveaux rapports** : Modifier `ReportGenerator`

## 🚨 Gestion d'erreurs

Le système est conçu pour **ne jamais planter** :

- **Fichiers corrompus** : Détection et récupération automatique
- **Encodages problématiques** : Fallback sur plusieurs encodages
- **Données manquantes** : Valeurs par défaut et logging détaillé
- **Ressources limitées** : Gestion de la mémoire et traitement par lots

## 📈 Performance

### Optimisations

- **Traitement par lots** : Configurable selon la mémoire disponible
- **Gestion mémoire** : Monitoring et optimisation automatique
- **Cache intelligent** : Réutilisation des calculs coûteux
- **Logging efficace** : Rotation automatique des fichiers de log

### Benchmarks

- **50 avis** : < 5 secondes
- **500 avis** : < 30 secondes  
- **5000 avis** : < 5 minutes

## 🤝 Contribution

### Standards de code

- **Style** : PEP 8 avec Black
- **Documentation** : Docstrings complètes
- **Tests** : Couverture > 80% pour les nouvelles fonctionnalités
- **Git** : Messages de commit descriptifs

### Processus de contribution

1. Fork du projet
2. Création d'une branche feature
3. Développement avec tests
4. Vérification qualité (Black + Flake8 + Tests)
5. Pull request avec description détaillée

## 📝 Changelog

### Version 1.0.0
- Analyse de sentiment VADER avec optimisations françaises
- Support complet des emojis et caractères spéciaux
- Génération de rapports JSON et CSV
- Interface en ligne de commande complète
- Gestion d'erreurs robuste
- Suite de tests complète (84 tests)

## 📄 Licence

Ce projet est développé dans le cadre d'un projet tutoré académique.

## 🆘 Support

### Problèmes courants

**Erreur d'encodage** :
```bash
# Vérifier l'encodage du fichier
file -i reviews.json
# Le système gère automatiquement UTF-8, Latin-1, CP1252
```

**Mémoire insuffisante** :
```bash
# Réduire la taille des lots dans config.json
{
  "processing": {
    "batch_size": 50
  }
}
```

**Résultats inattendus** :
```bash
# Mode verbeux pour diagnostic
python main.py reviews.json --verbose
```

### Contact

Pour toute question technique ou suggestion d'amélioration, consultez les logs détaillés générés par l'application ou utilisez le mode `--verbose` pour un diagnostic approfondi.