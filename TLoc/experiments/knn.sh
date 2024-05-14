#!/bin/bash

python -m localization.scripts.run_knn --metric manhattan

python -m localization.scripts.run_knn --metric euclidean

python -m localization.scripts.run_knn --metric euclidean --augmentation

python -m localization.scripts.run_knn --metric manhattan --ranking
