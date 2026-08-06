# Sentiment annotation guidelines

Reference for hand-labeling `data/processed/sentiment_annotation_sample.csv` (see `annotation.py`'s module docstring for why this sample exists and how it was drawn). The same definitions and rules are shown inline in `sentiment_annotation_tool.html` at the moment of each decision - this file is the durable, portfolio-facing copy.

## Labels

**Positive** - Réassurance, confiance, éloge

_optimisme affiché, remerciements, fierté, mise en avant d'un succès ou d'une bonne nouvelle_

**Neutral** - Procédural / informatif, sans valence affective nette

_réponse factuelle, annonce de calendrier, rappel de chiffres ou de procédure, renvoi vers un autre ministre sans jugement de valeur_

**Negative** - Critique, inquiétude, reproche

_mise en cause d'un adversaire ou d'une situation, alerte sur un problème, ton défensif ou combatif, déploration_

## Decision rules

- Juger le ton dominant de l'ensemble de la contribution, pas un mot isolé.
- En cas de mélange sans dominante claire, choisir le pôle le plus appuyé plutôt que de forcer un compromis - Neutre est réservé aux contributions sans contenu évaluatif du tout, pas aux contributions mitigées.
- L'ironie ou le sarcasme se codent selon le ton réel visé (souvent négatif/critique), pas selon le sens littéral des mots.
- Une déviation vers l'opposition sans reproche explicite reste Neutre ; avec un reproche explicite, elle devient Négatif.
- On code le sentiment exprimé PAR le Premier ministre, jamais le sentiment des autres à son sujet.
- En cas de mélange où un pôle est porté par des affirmations concrètes ou chiffrées et l'autre par une formule générique, privilégier le pôle concret - ex. un bilan chiffré pèse plus qu'un slogan de réassurance générique comme 'nous allons régler ça'.
- Si la contribution répond visiblement à une question absente du texte (le corpus ne contient que les tours du Premier ministre), coder uniquement son contenu explicite visible, sans deviner la question.
