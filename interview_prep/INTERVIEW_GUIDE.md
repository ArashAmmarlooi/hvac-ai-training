# Interview prep — plans HVAC / symboles / AI

**Ce que l’intervieweur montrait :** un plan technique (ex. gaines bleues = supply, rouges = return/exhaust, vert = équipement RTU/fan).  
**What the interviewer showed:** a technical drawing (e.g. blue ducts = supply, red = return/exhaust, green = RTU/fan equipment).

**Ce n’est pas “de la magie IA” dès le début :** souvent **vision classique** (couleur, formes, OCR) + parfois **ML** si les symboles varient.  
**It’s not “AI magic” at first:** often **classical vision** (color, shapes, OCR) + **ML** when symbols vary a lot.

---

## 1. Par où commencer le code / Where to start coding

| Étape | FR | EN |
|-------|----|----|
| 1 | Charger l’image (PDF → image si besoin) | Load image (rasterize PDF if needed) |
| 2 | Normaliser (rotation, échelle, contraste) | Normalize (rotation, scale, contrast) |
| 3 | **Légende** : extraire symboles + texte (OCR) | **Legend**: extract symbols + text (OCR) |
| 4 | **Plan** : détecter symboles (couleur / contours / template) | **Diagram**: detect symbols (color / contours / template) |
| 5 | Associer texte proche (distance, lignes de cote) | Link nearby labels (distance, leader lines) |
| 6 | API + file d’attente + statut job (Angular / .NET / Python) | API + queue + job status (Angular / .NET / Python) |

**Fichier démo :** `diagram_pipeline_demo.py`  
**ML simple :** `ml_basics_demo.py`

---

## 2. Quand utiliser quoi / When to use what

| Problème | Outil typique |
|----------|----------------|
| Lignes colorées fixes (bleu/rouge/vert) | OpenCV : masques HSV + contours |
| Légende stable, symboles identiques | Template matching |
| Symboles tournés / échelles différentes | Features (SIFT) ou **embeddings** / petit CNN |
| Beaucoup de types de symboles | **Object detection** (YOLO) ou classifieur CNN |
| Texte sur le plan | **OCR** (Tesseract, EasyOCR, Azure Read) |
| Doc interne + questions en langage naturel | **RAG** (pas pour remplacer la géométrie du plan) |

---

## 3. Phrase d’entretien (honnête) / Interview sentence (honest)

**FR :** « Pour un MVP, je segmente par couleur et forme, j’OCR la légende, je matche par template ou embedding, et je garde un score de confiance pour revue humaine. »  
**EN:** « For an MVP, I segment by color and shape, OCR the legend, match via template or embeddings, and keep confidence scores for human review. »

---

## 4. Architecture (comme ton interview) / Architecture (like your interview)

```
Angular upload → .NET API (job id, fichier) → queue → Python worker
     ↑                                              ↓
     └──────── poll / WebSocket ←── mise à jour statut + JSON résultats
```

**Statuts exemple :** `queued` → `preprocessing` → `legend_parsed` → `symbols_detected` → `done` / `failed`

---

## 5. ML / DL en 5 minutes / ML / DL in 5 minutes

- **ML classique (XGBoost, etc.) :** tu fabriques des **features** (colonnes) → modèle prédit une **classe** ou un **nombre**.  
- **Classical ML:** you build **features** (columns) → model predicts **class** or **number**.

- **Deep learning (CNN) :** le modèle apprend les features depuis les **pixels** (images).  
- **Deep learning (CNN):** the model learns features from **pixels** (images).

- **Entraînement :** données étiquetées → train/val split → métriques (precision, recall, F1) → même preprocessing en prod.  
- **Training:** labeled data → train/val split → metrics → **same preprocessing in production**.

---

## 6. Pratique cette semaine / Practice this week

1. `pip install -r requirements.txt`
2. `python diagram_pipeline_demo.py --image "chemin\vers\plan.png"`
3. `python ml_basics_demo.py`
4. Réécrire à voix haute le pipeline en 60 secondes (FR puis EN).
