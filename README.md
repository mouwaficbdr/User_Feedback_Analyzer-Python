![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NLP](https://img.shields.io/badge/NLP-Rule--based-orange.svg)


# UserFeedbackAnalyzer-Python 🇫🇷

Un moteur d'analyse de sentiment basé sur des règles linguistiques, conçu spécifiquement pour les **avis clients en français**. 
Contrairement aux modèles ML génériques qui sont lourds, cette solution utilise une **approche lexicale intelligente** combinée à des règles linguistiques un peu complexes (négation, intensification, expressions) pour atteindre une haute précision avec une utilisation minimale de ressources.

##  Vue d'ensemble

Ce moteur traite les avis clients en français pour les classifier en trois catégories : **Positif**, **Négatif** ou **Neutre**.
Il est conçu pour être **prêt pour la production** , en gros: robuste, rapide et facile à déployer sans dépendances GPU lourdes.

## Fonctionnalités Clés

- **Analyseur Français Intelligent** : Lexique intégré complet (~1000+ mots) spécifiquement ajusté pour les retours clients.
- **Intelligence Linguistique** :
  - **Gestion de la négation** : Interprète correctement "ce n'est pas bon" vs "c'est bon".
  - **Intensificateurs** : Distingue "bon" de "vraiment très bon".
  - **Détection d'expressions** : Reconnaît des expressions comme "ne marche pas", "hors de prix", "vaut le coup".
  - **Logique "Mais"** : Gère les mais en milieu de phrase qui suggère un contraste entre les deux bouts de phrases (ex: "Bon produit **mais** livraison lente").
- **Prêt pour la Production** :
  - **Gestion des erreurs** : Gestion d'erreurs robuste garantissant que le pipeline ne plante jamais sur des données incorrectes.
  - **Rapports Détaillés** : Génère des résumés complets (JSON) et des résultats clairs ligne par ligne (CSV).
  - **Performance** : Capable de traiter des milliers d'avis par seconde sur un CPU standard. (Normalement mdr)

## Démarrage

### Prérequis

- Python 3.8 ou supérieur
- pip

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/mouwaficbdr/UserFeedbackAnalyzer-Python.git
cd UserFeedbackAnalyzer-Python

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances (Léger !)
pip install -r requirements.txt
```

### Utilisation

```bash
# Analyser le fichier reviews.json par défaut
python main.py

# Analyser un fichier spécifique et sauvegarder dans un dossier
python main.py mes_avis.json --output-dir ./resultats

# Utiliser une configuration personnalisée
python main.py avis.json --config config.json
```

## Output

### 1. Rapport Récapitulatif (`summary.json`)
Fournit une vue d'ensemble de l'analyse (positifs, negatifs, neutres).

```json
{
  "analysis_summary": {
    "total_reviews": 50,
    "sentiment_distribution": {
      "positive": { "count": 12, "percentage": 24.0 },
      "negative": { "count": 9, "percentage": 18.0 },
      "neutral": { "count": 29, "percentage": 58.0 }
    }
  }
}
```

### 2. Résultats Détaillés (`results.csv`)
Un fichier CSV contenant chaque avis avec son score et sa classification.

```csv
review_id,review_text,sentiment_final,sentiment_score,processing_errors
REV001,"Excellent produit !",Positive,0.8516,
REV002,"Service décevant",Negative,-0.7269,
```

## Configuration & Seuils

La logique de classification repose sur un score adopté, entre **-1.0** (Négatif) et **+1.0** (Positif).

### Seuils par Défaut
- **Positif** : Score > **0.05**
- **Négatif** : Score < **-0.05**
- **Neutre** : Entre -0.05 et 0.05

**Justification** : Ces seuils sont choisis pour être légèrement inclusifs pour la neutralité. Dans les avis clients, un commentaire légèrement positif ("c'est ok") est souvent juste de la neutralité polie. Un petit tampon autour de 0.0 garantit que seul le texte vraiment avec une opinion est classé comme Positif ou Négatif.

## Architecture

```
UserFeedbackAnalyzer-Python/
├── src/
│   ├── analysis/          # Logique centrale (Analyseur Français Intelligent)
│   │   ├── sentiment_analyzer.py
│   │   └── french_lexicon.py  # Le "Cerveau" (Dictionnaire & Règles)
│   ├── engine.py          # Orchestrateur (Patron Façade)
│   ├── models/            # Structures de données (objet Review)
│   ├── data/              # Chargement & validation des données
│   └── reporting/         # Génération de sorties
├── tests/                 # Tests unitaires
└── main.py                # Point d'entrée CLI
```

## Tests

```bash
# Exécuter tous les tests
python -m pytest tests/ -v
```

## 📝 Licence

Licence MIT.
