## Process Review – Pre‑ and Post‑V2 Updates

### Scope
- **Pre‑V2 cells (0‑25)**: Original stroke modeling workflow covering EDA, baseline preprocessing, splits, PCA exploration, baseline classifiers, validation/test evaluation, interpretability, and diagnostics exported in `logs/stroke_modeling_outputs.txt` cells 3‑40.
- **Post‑V2 cells (≥26)**: Enhancements introduced today starting at the markdown marker `V2`, including new summaries, advanced feature engineering, imbalance mitigation experiments, threshold tuning, ensemble stacking, subgroup fairness diagnostics, and comparison tables (cells 44‑52 in the refreshed export).

### Pre‑V2 Highlights
1. **EDA & Feature Engineering**  
   - Loaded `Data/healthcare-dataset-stroke-data.csv`, created BMI-missing flag, senior indicator, log glucose, and age buckets.  
   - Ran summary stats, KDE/count plots, correlation heatmap, and hypothesis checks confirming age/chronic conditions drive stroke risk.
2. **Data Splits & Preprocessing**  
   - Stratified 60/20/20 split with similar prevalence (~4.9%).  
   - Built `ColumnTransformer` with median-imputed scaled numerics and OHE categoricals.
3. **Model Baselines**  
   - Explored PCA (PC1+PC2 ≈ 40% variance) without strong separation.  
   - Cross-validated logistic (class_weight balanced), random forest, gradient boosting. Logistic won on ROC AUC (~0.854) and recall (~0.78).
4. **Hyperparameter Tuning & Evaluation**  
   - Grid search on logistic PCA pipeline; best config used no PCA, C=0.1, L2.  
   - Validation/test metrics: ROC AUC ≈0.836, recall ≈0.82, precision ≈0.14; FP:TP ≈6:1 on test (scaling to 1,000 patients yields ~247 FP vs 40 TP).  
   - Generated ROC/PR/confusion plots and group diagnostics (gender, residence, age buckets) showing zero recall for <45 cohorts.
5. **Interpretability**  
   - Logged logistic coefficients aligning with medical expectations (age, age buckets, glucose, work type).  
   - RF feature importances highlighted similar drivers.

### Post‑V2 Enhancements
1. **Documentation & Context**  
   - Added `## Remediation Summary` and narrative of issues/next steps before new code, preserving historical transparency.
2. **Advanced Feature Engineering**  
   - Created interaction features (age×conditions, glucose×BMI, encoded age bucket stats) and built `advanced_preprocessor`.  
   - Oversampled under‑45 positives (`oversample_young_minority`) for targeted experimentation.
3. **Cost-Sensitive Modeling & Thresholding**  
   - Implemented `build_cost_sensitive_logistic`, helper metrics, and FP:TP-constrained threshold search.  
   - Best threshold (≈0.65) improved precision to 0.21 (FP:TP ≈3.7:1) but reduced recall to 0.14.
4. **Resampling & Ensembling**  
   - Built SMOTE logistic and Balanced RF pipelines within imbalanced-learn.  
   - Constructed stacking ensemble over cost-sensitive, SMOTE logistic, and balanced RF with meta-logistic regressor. Threshold tuned to hit FP:TP ≈1.5:1 (precision 0.40, recall 0.08).
5. **Age-Aware Model**  
   - Trained logistic on young-oversampled data and tuned thresholds: precision 0.31, recall 0.08 (FP:TP ≈2.25:1), improving attention to younger cohorts.
6. **Diagnostics & Logging**  
   - Group metrics now printed to text (captured in export) for cost-sensitive, age-aware, and stacked models.  
   - Comparison table summarises precision/recall/F1/balanced accuracy/FP:TP across baseline and new variants.
7. **Automation**  
   - Ensured notebook executes via `nbconvert` post-dependency install (`imbalanced-learn`), and reran `export_notebook_outputs.py` so `logs/stroke_modeling_outputs.txt` reflects every new cell.

### Findings & Next Steps
- Trade-offs remain: high recall requires tolerating ~6 FP per TP; tightening thresholds or stacking reduces FP burden but collapses recall (<0.15).
- Younger cohorts (<45) still receive zero or near-zero recall despite oversampling—feature signal may be insufficient.
- Recommended future work (per summary block):
  1. Consider focal-loss or gradient boosting variants with custom class weights/threshold optimization.
  2. Apply bucket-specific resampling (e.g., SMOTE-Tomek for younger groups) and probability calibration.
  3. Define acceptable FP:TP & recall targets with stakeholders and tune thresholds accordingly.
  4. Seek additional positive cases or external validation to improve generalizability.

This log should accompany the exported outputs so reviewers can trace when and why the V2 changes were introduced.

