# Transformer-Localization

Usage: `python -m localization.scripts.run_knn --metric euclidean`

KNN Experiment: `./experiments/knn.sh > experiments/results/knn.txt`

Train tLoc: `python -m localization.scripts.train_tloc`
Execute tLoc: `python -m localization.scripts.run_tloc`
Execute tLoc (different building/floor): `python -m localization.scripts.run_tloc -b 1 -f 2`

Filtering the data:

1- Train tLoc: `python -m localization.scripts.train_tloc`
2- Filter training data: `python -m localization.scripts.filter_tloc`
3- Train with filtered data: `python -m localization.scripts.train_tloc_filtered`
4- Execute tLoc as normal