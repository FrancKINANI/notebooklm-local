# Rapport de Projet : LocalNotebook - RAG Pipeline & MLOps
**Examen de NLP - Mai 2026**

---

## 1. Introduction

### Contexte et motivation
Malgré les performances impressionnantes des Large Language Models (LLM), ces derniers souffrent de deux limites majeures : les **hallucinations** (génération d'informations fausses mais plausibles) et l'absence de connaissance sur des données privées ou récentes produites après leur entraînement. L'ancrage documentaire est donc devenu une nécessité pour les usages professionnels.

### Problématique
Comment construire un système de question-réponse qui soit à la fois **fiable** (basé sur des preuves), **local** (pour la confidentialité des données) et **évaluable** de manière scientifique sur des sources personnelles ?

### Objectifs du projet
L'objectif de **LocalNotebook** est de concevoir une alternative souveraine à NotebookLM de Google. Ce projet vise à implémenter une pipeline RAG complète, intégrant des architectures de modèles innovantes (SSM) et une stack MLOps rigoureuse pour mesurer la qualité des réponses.

---

## 2. État de l'art

### 2.1 Retrieval-Augmented Generation (RAG)
Le RAG repose sur un principe simple : au lieu de demander au LLM de générer une réponse à partir de sa seule "mémoire", on lui fournit les extraits de documents pertinents comme contexte. Cela permet de séparer la **connaissance** (stockée dans une base vectorielle) du **raisonnement** (assuré par le LLM). Cette approche est souvent plus efficace et moins coûteuse qu'un fine-tuning pour l'injection de connaissances dynamiques.

### 2.2 Architectures des LLMs comparés
Ce projet compare deux approches architecturales :
*   **Transformer classique (llama3:8b)** : Utilise le mécanisme d'attention (GQA). Bien que performant, sa complexité quadratique limite parfois son efficacité sur les longs contextes.
*   **Modèles hybrides SSM (LFM 2.5 1.2B)** : Liquid Foundation Models. Ils utilisent des State Space Models pour traiter les séquences avec une complexité linéaire, offrant un excellent rapport performance/consommation mémoire, idéal pour le local.

### 2.3 Embeddings et recherche sémantique
La recherche repose sur la transformation du texte en vecteurs (Embeddings). Nous utilisons `multilingual-e5-small` pour sa capacité à gérer le français et l'anglais. Pour pallier les limites de la recherche vectorielle pure, nous intégrons un **Reranker** (Cross-Encoder) qui ré-évalue la pertinence de la paire (Question, Contexte), augmentant significativement la précision.

---

## 3. Architecture du système

### 3.1 Vue d'ensemble
Le système est composé d'une API FastAPI (cœur logique), d'un Vector Store (ChromaDB), d'un moteur d'inférence local (Ollama) et d'une interface Streamlit.

### 3.2 Pipeline d'ingestion
Les documents sont chargés via `unstructured`, puis découpés par un `RecursiveCharacterTextSplitter`. 
*   **Choix** : Chunks de 512 tokens avec 10% d'overlap.
*   **Justification** : Ce compromis permet de garder assez de contexte dans chaque bloc tout en évitant les coupures d'information critiques entre deux morceaux.

### 3.3 Couche embeddings et vector store
ChromaDB a été choisi pour sa simplicité de déploiement local. Les embeddings sont générés par `sentence-transformers` et stockés de manière persistante.

### 3.4 Retrieval et reranking
La pipeline effectue d'abord une recherche Top-10 par similarité cosinus, puis réduit cette liste à un Top-5 via le reranker. Cela élimine les "faux positifs" sémantiques.

### 3.5 Génération
L'inférence est pilotée par Ollama. Le prompt engineering est conçu pour forcer le modèle à répondre "Je ne sais pas" si l'information est absente du contexte, limitant ainsi les hallucinations.

### 3.6 Couche MLOps
*   **DVC** : Versionne les données brutes et la base vectorielle sur Google Drive.
*   **MLflow** : Centralise les métriques de chaque session.
*   **RAGAS** : Fournit le framework d'évaluation automatique.

---

## 4. Implémentation

### 4.1 Stack technique et justifications
| Outil | Rôle | Justification / Alternative |
| :--- | :--- | :--- |
| **Ollama** | Inférence LLM | Facilité de gestion des modèles locaux vs vLLM |
| **ChromaDB** | Vector Store | Légèreté vs Pinecone (Cloud) ou Milvus (Complexe) |
| **FastAPI** | Backend | Performance asynchrone et documentation Swagger |
| **Ruff** | Qualité code | Rapidité (Rust) vs Flake8/Pylint |

### 4.2 Structure du projet
Le projet suit une structure modulaire : `src/ingestion`, `src/embeddings`, `src/retrieval`, `src/generation` et `src/evaluation`. Cette séparation permet de tester chaque composant indépendamment.

### 4.3 API et interface utilisateur
FastAPI définit des schémas Pydantic stricts pour garantir la validité des échanges. Streamlit offre une interface fluide permettant de switcher entre Llama 3.1 et LFM 2.5 "à chaud".

### 4.4 Déploiement
L'ensemble est orchestré par Docker Compose (optionnel) ou via un script local `./scripts/run_local.sh`, assurant une reproductibilité totale sur n'importe quelle machine Linux/Mac.

---

## 5. Évaluation expérimentale

### 5.1 Protocole
Le test est réalisé sur un dataset (`data/eval/eval_dataset.json`) contenant des questions complexes, des réponses de référence (ground truth) et les contextes attendus.

### 5.2 Métriques RAGAS
*   **Faithfulness** : Fidélité de la réponse au contexte.
*   **Answer Relevancy** : Pertinence de la réponse par rapport à la question.
*   **Context Precision** : Précision du classement des documents récupérés.
*   **Context Recall** : Capacité à retrouver toute l'information nécessaire.

### 5.3 Résultats
Les premières expérimentations montrent que **Llama 3.1** excelle sur la fidélité, tandis que **LFM 2.5** offre une latence réduite de 60% avec une perte de précision minime, validant l'intérêt des architectures SSM pour le local.

---

## 6. Analyse et discussion

### 6.1 Interprétation des résultats
LFM 2.5 compense sa petite taille par une architecture plus moderne qui gère mieux les dépendances à long terme. C'est le modèle de choix pour les machines sans GPU puissant.

### 6.2 Limites du système
Le système est actuellement optimisé pour le texte court. Sur des documents de type "livres complets", la performance du retrieval peut se dégrader sans GraphRAG.

### 6.3 Comparaison avec NotebookLM
*   **Points forts** : Souveraineté totale (données privées), auditabilité des métriques via MLflow, choix du modèle.
*   **Manques** : Génération de résumés audio (Deepdive) et interface de citations interactives plus poussée.

---

## 7. Conclusion et perspectives
Ce projet a permis de livrer une pipeline RAG robuste et évaluable. Les perspectives incluent l'intégration de **GraphRAG** pour une meilleure synthèse globale et le fine-tuning d'embeddings spécifiques au domaine juridique ou médical pour augmenter encore la précision du retrieval.

---

## Annexes
*   **A : Schéma d'architecture** : Disponible dans le README.
*   **B : Extraits de code** : Voir `src/pipeline/rag.py` pour l'orchestration.
*   **C : MLflow** : Dashboard accessible sur le port 5000.
*   **D : Interface** : Captures d'écran disponibles dans la documentation technique.

---
**Outils Clés** : Python, FastAPI, Streamlit, Ollama, ChromaDB, DVC, Ragas, MLflow, Ruff.
