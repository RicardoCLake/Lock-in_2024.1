# Appendix: Model Hyperparameters

| Model | Hyperparameters |
| :--- | :--- |
| **XGBoost** | `n_estimators=100`, `max_depth=6`, `random_state=42`, `booster='gbtree'` |
| **WKNN_Pos_Euclid** | `n_neighbors=5`, `weights='distance'`, `metric='euclidean'`, `Representation=Positive` |
| **WKNN_Pow_Euclid** | `n_neighbors=5`, `weights='distance'`, `metric='euclidean'`, `Representation=Powered (e)` |
| **WKNN_Pos_Sorensen** | `n_neighbors=5`, `weights='distance'`, `metric='braycurtis'` (Sorensen), `Representation=Positive` |
| **WKNN_Pow_Sorensen** | `n_neighbors=5`, `weights='distance'`, `metric='braycurtis'` (Sorensen), `Representation=Powered (e)` |
| **TLoc (RBF)** | Default configuration defined inside python script class |
