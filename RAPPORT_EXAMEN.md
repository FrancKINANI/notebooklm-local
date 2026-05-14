# Rapport de Projet : LocalNotebook - RAG Pipeline & MLOps
**Examen de NLP - Mai 2026**

## 1. Introduction
Ce projet, baptisé **LocalNotebook**, est une implémentation avancée d'un système de **Retrieval-Augmented Generation (RAG)** conçu pour fonctionner de manière totalement locale et privée. L'objectif est de fournir une alternative souveraine à des outils comme NotebookLM, permettant l'ingestion de documents personnels et l'interrogation via des modèles de langage (LLM) performants, tout en intégrant une boucle de rétroaction MLOps pour l'évaluation continue.

## 2. Architecture Technique
Le système repose sur un pipeline RAG modulaire décomposé en quatre étapes clés :

1.  **Ingestion & Chunking** : Utilisation de `unstructured` pour le parsing de fichiers (PDF, HTML) et d'une stratégie de découpage récursif (`RecursiveCharacterTextSplitter`) pour maintenir la cohérence contextuelle.
2.  **Embeddings & Vector Store** : Transformation des textes en vecteurs via le modèle `multilingual-e5-small` et stockage dans **ChromaDB**.
3.  **Retrieval & Reranking** : Recherche sémantique par similarité cosinus, suivie d'une étape de **Reranking** via un `Cross-Encoder` (`ms-marco-MiniLM-L-6-v2`) pour affiner la pertinence des sources.
4.  **Génération** : Inférence locale via **Ollama**, supportant le switch dynamique entre deux architectures de modèles :
    *   **Llama 3 (8B)** : Architecture Transformer classique (GQA).
    *   **LFM 2.5 (1.2B)** : Architecture hybride basée sur les State Space Models (SSM), offrant une efficacité mémoire supérieure pour les appareils locaux.

## 3. Stack MLOps & Évaluation
L'originalité du projet réside dans son intégration MLOps pour garantir la qualité des réponses :

*   **Évaluation Automatisée (RAGAS)** : À chaque ingestion, une tâche de fond calcule des métriques "LLM-as-a-judge" :
    *   *Faithfulness* : La réponse est-elle fidèle aux sources ?
    *   *Answer Relevancy* : La réponse répond-elle vraiment à la question ?
    *   *Context Precision/Recall* : La recherche documentaire est-elle efficace ?
*   **Expérimentation (MLflow)** : Toutes les métriques et paramètres (taille des chunks, top-k) sont loggués dans **MLflow**, permettant une comparaison objective des modèles et des configurations.
*   **Versionnement (DVC)** : Les données et les bases vectorielles sont versionnées et synchronisées sur un stockage distant (**Google Drive**), assurant la reproductibilité complète des expériences.

## 4. Conclusion
LocalNotebook démontre qu'il est possible de construire un système de NLP complexe, performant et auditable sans dépendre d'API cloud tierces. L'utilisation d'outils comme `uv` pour la gestion des dépendances et `Docker` pour l'orchestration souligne une approche logicielle rigoureuse, prête pour un déploiement en production.

---
**Outils Clés** : Python, FastAPI, Streamlit, Ollama, ChromaDB, DVC, Ragas, MLflow.
