# Author: Alexandre Maranhão <alexandremr01@gmail.com>

import pandas as pd
from os import path
import numpy as np
from sklearn.model_selection import train_test_split

from localization.dataset.dataset import Dataset


def load_ujiindoor_loc(data_folder='data', transform=True):
    df_training = pd.read_csv(path.join(data_folder, 'trainingData.csv'))
    df_validation = pd.read_csv(path.join(data_folder, 'validationData.csv'))
    return UJIIndoorLoc(df_training, df_validation, transform=transform)

def calculate_means(dataframe):
    return [dataframe['LATITUDE'].mean(), dataframe['LONGITUDE'].mean()]

class UJIIndoorLoc(Dataset):
    def __init__(self, df_training, df_validation, transform):
        self.df_training, self.df_validation, self.input_columns = self.preprocess(df_training, df_validation,
                                                                                   transform)
        self.location_columns = ['LONGITUDE', 'LATITUDE', 'FLOOR', 'BUILDINGID']

    def preprocess(self, df_training, df_validation, transform=True, mix_datasets=False):

        if mix_datasets and transform:
            df = pd.concat([df_training, df_validation])
            df_training, df_validation = train_test_split(df, test_size=0.2, random_state=0)

        input_columns = list(df_training.filter(regex='WAP\d+').columns)

        df_training = self.drop_rows_without_information(df_training, input_columns)
        df_validation = self.drop_rows_without_information(df_validation, input_columns)
        if transform:
            df_training = self.transform(df_training, input_columns)
            df_validation = self.transform(df_validation, input_columns)

            buildings = df_training['BUILDINGID'].unique()
            for building in buildings:
                building_indexes_train = df_training['BUILDINGID']==building
                building_indexes_validation = df_validation['BUILDINGID']==building
                means = calculate_means(df_training[building_indexes_train])
                df_training.loc[building_indexes_train,'x'] = df_training[building_indexes_train]['LATITUDE'] - means[0]
                df_training.loc[building_indexes_train,'y'] = df_training[building_indexes_train]['LONGITUDE'] - means[1]
                df_validation.loc[building_indexes_validation,'x'] = df_validation[building_indexes_validation]['LATITUDE'] - means[0]
                df_validation.loc[building_indexes_validation,'y'] = df_validation[building_indexes_validation]['LONGITUDE'] - means[1]

        # drop columns without training information
        mask = df_training[input_columns] != 0
        number_of_samples = np.sum(mask, axis=0)
        columns_without_information_names = list(number_of_samples.index[number_of_samples == 0])
        df_training = df_training.drop(columns=columns_without_information_names)
        df_validation = df_validation.drop(columns=columns_without_information_names)

        input_columns = list(df_training.filter(regex='WAP\d+').columns)

        return df_training, df_validation, input_columns

    def drop_rows_without_information(self, df, input_columns):
        mask = df[input_columns] != 100
        number_of_aps = np.sum(mask, axis=1)
        indexes_without_information = number_of_aps.index[number_of_aps == 0]
        clean_df = df.drop(indexes_without_information).reset_index(drop=True)
        return clean_df

    def transform(self, df, input_columns):
        # Sanity check
        if any(df[input_columns[0]] == 100):
            df = df.replace(100, -105)
            df[input_columns] = df[input_columns] + 105
        return df.copy()

    def get_X(self):
        return self.df_training[self.input_columns].to_numpy(), self.df_validation[self.input_columns].to_numpy()

    # TODO: deprecate in favor of getX
    def get_X_df(self):
        return self.df_training[self.input_columns], self.df_validation[self.input_columns]

    def get_full_df(self):
        return self.df_training

    def get_categorical_y(self):
        y_train = np.hstack([(self.df_training['BUILDINGID']).to_numpy().reshape(-1, 1),
                             (self.df_training['FLOOR']).to_numpy().reshape(-1, 1)])
        y_test = np.hstack([(self.df_validation['BUILDINGID']).to_numpy().reshape(-1, 1),
                            (self.df_validation['FLOOR']).to_numpy().reshape(-1, 1)])
        return y_train, y_test

    def get_floor_data(self, building=None, floor=None, phoneid=None, reset_means=True):
        df_train = self.df_training
        df_test = self.df_validation
        if building is not None:
            df_train = df_train[df_train['BUILDINGID'] == building]
            df_test = df_test[df_test['BUILDINGID'] == building]
            building_means = calculate_means(df_train)
        if floor is not None:
            df_train = df_train[df_train['FLOOR'] == floor]
            df_test = df_test[df_test['FLOOR'] == floor]
        if phoneid is not None:
            df_train = df_train[df_train['PHONEID'] == phoneid]
            df_test = df_test[df_test['PHONEID'] == phoneid] 

        return UJIIndoorLoc(df_train, df_test, transform=False)

    def get_normalized_y(self):
        """
        Converts Latitude and Longitude to a reference with mean (x,y) = 0

        Args:
            -df (DataFrame): input DataTrame
            -mean (float[2]): optional mean to calculate new reference

        Returns:
            -df (DataFrame): output DataFrame
            -mean (float[2]): mean used to calculate reference

        """
        normalized_y_train = self._get_normalized_df(self.df_training)
        normalized_y_test = self._get_normalized_df(self.df_validation)
        return normalized_y_train, normalized_y_test

    def get_continuous_y(self):
        continuous_columns = ['LATITUDE', 'LONGITUDE']
        return self.df_training[continuous_columns].to_numpy(), self.df_validation[continuous_columns].to_numpy()

    def _get_normalized_df(self, df):
        """
        Converts Latitude and Longitude to a reference with mean (x,y) = 0

        Args:
            -df (DataFrame): input DataTrame
            -mean (float[2]): optional mean to calculate new reference

        Returns:
            -df (DataFrame): output DataFrame
            -mean (float[2]): mean used to calculate reference

        """
        return df[['x','y']]
