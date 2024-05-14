import argparse
import random
import time

import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier

from localization import dataset
from localization import augmentation
from localization.transformations.ranking_based import RankingBased
from localization import utils


def evaluate_continuous_location_model(discrete_location_model, continuous_location_model, x_test, y_continuous_test):
    start_time = time.time()
    predicted_discrete_location = discrete_location_model.predict(x_test)
    predicted_continuous_location = np.zeros((x_test.shape[0], 2))

    locations = np.unique(predicted_discrete_location, axis=0)
    # for each discrete location, apply the respective model
    for location in locations:
        indexes = filter_matrix_by_row(predicted_discrete_location, location)
        x_test_in_loc = x_test[indexes]
        predicted_continuous_location[indexes, :] = continuous_location_model[tuple(location)].predict(x_test_in_loc)

    end_time = time.time()
    distances = np.linalg.norm(predicted_continuous_location - y_continuous_test, axis=1)
    elapsed_seconds = (end_time - start_time)
    return distances, elapsed_seconds


def evaluate_categorical_location_model(model, x_test, y_test):
    start_time = time.time()
    y_hat = model.predict(x_test)
    end_time = time.time()

    correct_indexes = np.argwhere(((y_hat == y_test).sum(axis=1) == 2)).flatten()
    score = correct_indexes.shape[0] / y_hat.shape[0]

    elapsed_seconds = (end_time - start_time)
    return score, correct_indexes, elapsed_seconds


def filter_matrix_by_row(y, target_y):
    """
    :param y: matrix N x O
    :param target_y: vector 1 x O
    :return: indexes where y matches target_y
    """
    match = (y == target_y)
    row_equal = np.all(match, axis=1)
    indexes = np.where(row_equal)
    return indexes


def train_discrete_location_model(x_train, y, metric, k):
    neigh = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
    classifier = MultiOutputClassifier(neigh, n_jobs=-1)
    classifier.fit(x_train, y)
    return classifier


def train_continuous_location_models(x_train, y_train, y_continuous_train, metric, k):
    locations = np.unique(y_train, axis=0)
    models = {}
    for location in locations:
        # create a filter for this location
        indexes = filter_matrix_by_row(y_train, location)
        x_train_in_loc = x_train[indexes]
        y_continuous_train_in_loc = y_continuous_train[indexes]
        # create a model to learn predict the continuous in this discrete location
        neigh = KNeighborsRegressor(n_neighbors=k, metric=metric)
        neigh.fit(x_train_in_loc, y_continuous_train_in_loc)
        models[tuple(location)] = neigh
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run KNN')
    parser.add_argument('--metric', type=str, choices=['euclidean', 'manhattan'], default='euclidean')
    parser.add_argument('--seed', type=int, default=33, help='Random seed')
    parser.add_argument('--ranking', action='store_true')
    parser.add_argument('--augmentation', action='store_true')
    args = parser.parse_args()

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    utils.make_deterministic(args.seed)

    dataset = dataset.load_ujiindoor_loc()
    x_train, x_test = dataset.get_X()
    y_train, y_test = dataset.get_categorical_y()
    y_continuous_train, y_continuous_test = dataset.get_continuous_y()

    if args.augmentation:
        pa = augmentation.PowerAverages(dataset)
        new_x, new_y, new_y_continuous = pa.get_augmented_data(new_points_per_location=10)
        x_train = np.vstack([x_train, new_x])
        y_train = np.vstack([y_train, new_y])
        y_continuous_train = np.vstack([y_continuous_train, new_y_continuous])

    score, elapsed_time = 0, 0
    if args.ranking:
        ranking_based = RankingBased()
        # default value relates with how we treat nulls in the ranking
        default_value = x_train.shape[1]
        x_train = ranking_based.transform(x_train, default_value=default_value)
        x_test = ranking_based.transform(x_test, default_value=default_value)

    discrete_location_model = train_discrete_location_model(x_train, y_train, args.metric, 3)
    continuous_location_models = train_continuous_location_models(x_train, y_train, y_continuous_train, args.metric, 3)

    score, _, elapsed_time = evaluate_categorical_location_model(discrete_location_model, x_test, y_test)
    print('Building and floor accuracy:', np.round(100 * score, 2))
    print('Prediction time:', np.round(elapsed_time, 2), 's\n')

    distances, elapsed_time = evaluate_continuous_location_model(discrete_location_model, continuous_location_models,
                                                                 x_test, y_continuous_test)
    print(f'Mean error = {np.mean(distances):.2f}, median error = {np.median(distances):.2f}, '
          f'P90 = {np.percentile(distances, 90):.2f}, P95 = {np.percentile(distances, 95):.2f}')
    print('Prediction time:', np.round(elapsed_time, 2), 's')
    print('End of execution')
