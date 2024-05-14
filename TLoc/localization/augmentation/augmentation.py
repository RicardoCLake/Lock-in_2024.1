# Author: Alexandre Maranhão <alexandremr01@gmail.com>
import numpy as np
import pandas as pd

class Augmentation:
    def __init__(self):
        pass


class PowerAverages:
    def __init__(self, dataset):
        """
        Initializes a PowerAverages object.

        Args:
            -dataset (Dataset)

        Attributes:
            -dataset (Dataset): keeps a reference to the dataset
            -power_averages_per_position (pd.DataFrame): matrix of power averages with one row per unique position and
            one column per visible router.
        """
        self.dataset = dataset
        self.distinct_positions = dataset.df_training[dataset.location_columns].drop_duplicates().reset_index(drop=True)
        self.power_averages_per_position = self.get_power_averages_per_position()
        self.power_stds_per_position = self.get_power_std_per_position()

    def get_power_averages_per_position(self):
        """Returns a matrix of power averages of size MxN where M is number of unique positions and N are the APs that
        are visible in any position. Note that not all APs will be in the return, since not all of them appear in all
        measurements of any position."""
        position_p_avg_map = self.distinct_positions.apply(
            lambda row: self._get_aps_for_position(row['LONGITUDE'], row['LATITUDE'], row['FLOOR']).mean(),
            axis=1
        )
        final_df = pd.DataFrame(columns=self.dataset.input_columns)
        final_df[position_p_avg_map.columns] = position_p_avg_map
        return final_df

    def get_power_std_per_position(self):
        """Returns a matrix of power standard deviations of size MxN where M is number of unique positions and N are the APs that
        are visible in any position. Note that not all APs will be in the return, since not all of them appear in all
        measurements of any position."""
        # for each position
        position_p_avg_map = self.distinct_positions.apply(
            lambda row: self._get_aps_for_position(row['LONGITUDE'], row['LATITUDE'], row['FLOOR']).std(),
            axis=1
        )
        final_df = pd.DataFrame(columns=self.dataset.input_columns)
        final_df[position_p_avg_map.columns] = position_p_avg_map
        return final_df

    def _get_aps_for_position(self, lon, lat, floor):
        """ Given a position, returns the RSS of each AP that is present in all measurements taken in that
        same position."""
        df = self.dataset.df_training
        # get all points in this position
        points_in_pos = df[(df['LONGITUDE'] == lon) & (df['LATITUDE'] == lat) & (df['FLOOR'] == floor)]
        points_in_pos = points_in_pos.loc[:, points_in_pos.columns.str.startswith("WAP")]
        # filter: select only aps that appear in all measurements. can be other rule.
        ap_filter = (~points_in_pos.isin([0])).sum() == points_in_pos.shape[0]
        filtered_aps_names = [key for key, value in ap_filter.items() if value]
        filtered_aps = points_in_pos[filtered_aps_names]
        return filtered_aps

    def get_augmented_data(self, new_points_per_location=1):
        new_x, new_y, new_y_continuous = [], [], []
        for i in range(new_points_per_location):
            new_x.append( np.random.normal(
                self.power_averages_per_position.loc[:, self.dataset.input_columns].fillna(0),
                self.power_stds_per_position.loc[:, self.dataset.input_columns].fillna(0)
            ))
            new_y.append(self.distinct_positions[['BUILDINGID', 'FLOOR']].to_numpy())
            new_y_continuous.append(self.distinct_positions[['LATITUDE', 'LONGITUDE']])
        return np.vstack(new_x), np.vstack(new_y), np.vstack(new_y_continuous)
