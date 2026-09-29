# Crop Recommendation Classifier

Educational, zero-cost ML prototype that predicts which crop is best suited to a
given set of soil and climate readings.

## Problem statement

Classify whether environmental and soil conditions are suitable for selected crop
categories (22 crops), given nitrogen (N), phosphorus (P), potassium (K),
temperature, humidity, soil pH, and rainfall.

## Dataset

**Crop Recommendation Dataset** — 2200 rows, 7 numeric features, 22 crop classes
(100 balanced samples each), no missing values, no duplicates.

- Source: Kaggle, "Crop Recommendation Dataset", uploaded by Atharva Ingle
  (https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).
  The dataset description credits augmentation from Indian rainfall, climate and
  fertilizer/soil datasets. Distributed under Kaggle's terms; used here strictly
  for non-commercial, educational purposes.
- Local copy used by this project: `data/Crop_recommendation.csv`.

## Repository layout

```
crop_project/
├── data/Crop_recommendation.csv        # dataset
├── train.py                            # full pipeline: EDA -> train -> evaluate -> save
├── notebook/crop_recommendation_analysis.ipynb   # narrated analysis notebook
├── app/app.py                          # Streamlit demo
├── models/                             # trained model + preprocessing objects (joblib)
├── figures/                            # EDA & evaluation plots (PNG)
├── report/                             # JSON metrics + classification report
├── requirements.txt
└── README.md
```

## Method summary

1. **Data quality** — checked for missing values, duplicates, class balance, and
   outliers (IQR rule). Dataset is clean and perfectly balanced; flagged outliers
   are genuine agronomic variation (e.g. rice's high rainfall need), not errors,
   so none were removed.
2. **EDA** — class distribution, feature histograms, correlation matrix, boxplots.
   P and K are moderately correlated (r ≈ 0.74); no feature encodes the label, so
   there is no data leakage.
3. **Preprocessing** — label-encoded the target; 80/20 stratified train/test
   split; `StandardScaler` fit for the scale-sensitive baseline (the tree model
   uses raw features).
4. **Modeling** —
   - Baseline: Logistic Regression (multinomial) → **97.3%** test accuracy.
   - Main model: Random Forest (300 trees) → **99.3%** test accuracy,
     **99.5% ± 0.3%** under 5-fold stratified cross-validation.
5. **Evaluation** — per-class precision/recall/F1 (`report/classification_report.json`),
   confusion matrix (`figures/confusion_matrix.png`).
6. **Interpretation** — Random Forest feature importances: rainfall and humidity
   dominate, followed by potassium and phosphorus, temperature, nitrogen and pH.
7. **Error analysis** — only 3/440 test samples misclassified, all between
   agronomically similar crops (e.g. lentil vs. mothbeans).

## Reproduce

```bash
pip install -r requirements.txt
python train.py          # regenerates figures/, report/, and models/
```

## Run the demo app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app loads `models/rf_crop_model.joblib` and `models/label_encoder.joblib`
(built by `train.py`), takes soil/climate sliders as input, and returns the
recommended crop plus a probability breakdown of the top-5 candidates.

## Results at a glance

| Model               | Test accuracy | Macro F1 | 5-fold CV accuracy |
|---------------------|:---:|:---:|:---:|
| Logistic Regression | 97.3% | 0.972 | — |
| Random Forest        | 99.3% | 0.993 | 99.5% ± 0.3% |

## Scope & constraints

Built entirely with free/open-source tools (pandas, scikit-learn, matplotlib,
seaborn, Streamlit) — no paid APIs or commercial platforms. This is an
educational prototype, not a production agricultural decision system; real
deployments would need region-specific validation, soil-testing protocols, and
agronomist review.

## Deliverables in this package

- Source code 
- Notebook 
- Trained model 
- `requirements.txt`
- README (this file)
- Streamlit application 
- Technical paper 
- Presentation 
