# Qualification Equipe Projet Tutoré

# Moteur d'Analyse de Sentiment

Durée du Challenge :  2 semaines.

Langage Principal Requis : Python 3.x (solution 100% open source).

Objectif Final : Concevoir une solution Python complète pour ingérer, analyser le sentiment d'un large corpus d'avis clients, et générer un rapport synthétique.

### Critère d'Évaluation Principal : Autonomie Architecturale

‼️Ce test se rapproche beaucoup d'un concours. Dans un concours, le niveau d'excellence différencie deux projets fonctionnels. Comprenez donc que dans un concours, dans un souci de départager, les détails et l'excellence comptent.

## I. Données et Contraintes Initiales

| **Élément**                    | **Description**                                                                                                                                                                                                                 |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Fichier d'entrée**           | Un fichier nommé `reviews.json` contenant 50 avis. Les champs incluent au minimum `review_id` et `review_text`. Disponible en pièce jointe tout en bas.                                                                         |
| **Contraintes de Données**     | Le programme doit être **robuste**. Le jeu de données inclura volontairement des cas problématiques : lignes vides, encodage non standard, avis très courts, ou caractères spéciaux. Votre solution ne doit **jamais crasher**. |
| **Contrainte d'Environnement** | Le projet doit être lancé via un simple appel sur la ligne de commande et fonctionner dans un environnement virtuel (l'utilisation de `requirements.txt` est obligatoire).                                                      |

## II. Exigences Fonctionnelles (Les Résultats Attendus)

Votre programme doit exécuter un pipeline complet et produire une sortie claire.

### Analyse de Sentiment

Le moteur doit classer chaque avis en l'une des trois catégories : **Positif**, **Négatif**, ou **Neutre**. Vous devez justifier dans votre documentation le seuil de décision choisi.

### Rapport de Synthèse

Un fichier de rapport simple (`summary.txt` ou `.json`) doit être généré, affichant clairement :

1. Le nombre total d'avis analysés.
2. Le pourcentage et le nombre absolu de Positifs, Négatifs et Neutres.

### Fichier de Sortie Détaillé

Un second fichier de sortie (`results.csv`) doit être créé, reprenant le contenu du fichier original, mais en ajoutant une entrée suppémentaire pour chaque review nommée `sentiment_final` avec le résultat de l'analyse pour chaque avis.

## III. Critères d'Évaluation (Ce qui fera la différence)

L'évaluation portera sur les indicateurs de qualité suivants:

### 1.  Architecture et Design Logiciel (40% de la Note)

* **Modularité et POO** 
* **Séparation des Responsabilités&#x20;**
* **Configuration**

### 2.  Bonnes Pratiques et Autonomie (30% de la Note)

* **Historique Git**
* **Documentation de Code**
* **Style** 
* **Recherche**

### 3. Robustesse et Qualité (30% de la Note)

* **Gestion des Erreurs** 
* **Traçabilité**
* **(Bonus) Tests Unitaires** 

  BONNE CHANCE À TOUS. LE DÉLAI EST DE 2 SEMAINES ET LE CHRONO DÉMARRE LE **20/10/2025.**

[reviews.json](https://app.affine.pro/workspace/08c37c9c-3b98-475b-a7cc-8d04b30cafb8/PQ4n9fu2TuRhE5SjqMAQA)
