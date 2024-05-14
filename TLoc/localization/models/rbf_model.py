import numpy as np
from scipy.interpolate import RBFInterpolator, Rbf
from scipy.special import erf, gamma, hyp2f1, expit
import matplotlib.pyplot as plt
import numba
from numba import jit, njit
from tqdm import tqdm
import scipy


@jit(nopython=True)
def pointinpolygon(x, y, poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = poly[0]
    for i in numba.prange(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit(parallel=True)
def verify_if_the_points_are_inside_the_polygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean)
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i, 0], points[i, 1], polygon)
    return D


class Model:

    def __init__(self,
                 num_buildings: int,
                 num_floors_in_each_building: dict,
                 path_to_the_polygon_files_of_each_building: dict,
                 sigma: float,
                 num_samples_per_ap: int
                 ):

        self.sigma = sigma
        self.num_samples_per_ap = num_samples_per_ap
        self.num_buildings = num_buildings
        self.num_floors_in_each_building = num_floors_in_each_building
        self.path_to_the_polygon_files_of_each_building = path_to_the_polygon_files_of_each_building

        self.building_polygons = {}
        self.building_polygons_holes = {}
        self.power_probability_masks = {}

        self.x_building = {}
        self.y_building = {}
        self.power_prior_probability_distribution = {}

        self.trained = False

        for building in range(self.num_buildings):
            self.building_polygons[building] = np.load(self.path_to_the_polygon_files_of_each_building[building]).T
            try:
                self.building_polygons_holes[building] = np.load(self.path_to_the_polygon_files_of_each_building[building].replace('.npy', '_hole.npy')).T
            except FileNotFoundError:
                self.building_polygons_holes[building] = None
            self.power_probability_masks[building] = {}
            self.power_prior_probability_distribution[building] = {}

        for building in range(self.num_buildings):
            for floor in range(self.num_floors_in_each_building[building]):
                self.power_probability_masks[building][floor] = {}
                self.power_prior_probability_distribution[building][floor] = {}

        self.eps = 1e-5
        self.grid_size_in_each_dimension_in_each_building = 100
        self.t_score_alpha = 0.05
        self.non_null_minimum_percentage = 0.10
        self.random_position_error = 0.01  # meters
        self.building_map_margin = 15  # meters
        self.max_power = 106

    def cumulative_distribution_function_of_t_student(self, x, v):

        return 0.5 + x * gamma((v + 1) / 2) * hyp2f1(1 / 2, (v + 1) / 2, 3 / 2, -(x ** 2) / v) / (
                np.sqrt(v * np.pi) * gamma(v / 2))

    def cumulative_distribution_function_of_power(self, power, mu, phi):

        v = np.ceil(self.num_samples_per_ap * (1 - phi) - 1)
        v = np.where(v <= 0, 1, v)
        cdf = phi * np.heaviside(power, 1) + (1 - phi) * self.cumulative_distribution_function_of_t_student(
            (power - mu) / self.sigma, v)

        return cdf

    def approximate_position_density_function_given_router(self, power, mu, phi):

        v = np.ceil(self.num_samples_per_ap * (1 - phi) - 1)
        v = np.where(v <= 0, 1, v)
        t_score = scipy.stats.t.ppf(0.5 + self.t_score_alpha, v)

        density_function = self.cumulative_distribution_function_of_power(power + t_score * self.sigma, mu,
                                                                          phi) - self.cumulative_distribution_function_of_power(
            power - t_score * self.sigma, mu, phi)
        density_function = density_function / density_function.sum()

        return density_function

    def get_all_routers_in_this_floor(self, columns):
        routers = []
        for column in columns:
            if 'WAP' in column:
                routers.append(column)
        return routers

    def radial_log_basis_function(self, r):
        return np.log(r + self.eps)

    def checking_non_null_minimum_percentage_of_samples(self, X_train, router):

        num_non_null_samples = (X_train[router] != 0).sum()
        num_total_samples = X_train[router].shape[0]
        non_null_percentage = num_non_null_samples / num_total_samples

        return (non_null_percentage > self.non_null_minimum_percentage)

    def get_mu_and_phi_estimation(self, X_train, router):

        phi_ = {}
        mu_ = {}

        for index in X_train.groupby(['x', 'y']).mean().index:
            x, y = index
            data_train = X_train[(X_train['x'] == x) & (X_train['y'] == y)]
            num_null_samples = (data_train[router] == 0).sum()
            num_total_samples = data_train.shape[0]
            phi_[index] = num_null_samples / num_total_samples

            nonzero_data_train = data_train[data_train[router] != 0]
            mu_[index] = nonzero_data_train[router].sum() / nonzero_data_train.shape[0] if nonzero_data_train.shape[
                                                                                               0] > 0 else 0

        X_train['phi'] = X_train.apply(lambda series: phi_[(series.x, series.y)], axis=1)
        X_train['mu'] = X_train.apply(lambda series: mu_[(series.x, series.y)], axis=1)

        X_train['x'] = X_train['x'] + np.random.normal(0, self.random_position_error, size=X_train['x'].shape[0])
        X_train['y'] = X_train['y'] + np.random.normal(0, self.random_position_error, size=X_train['y'].shape[0])

        return X_train

    def construct_building_map(self, dataset, building):

        filtered_dataset = dataset.get_floor_data(building=building)
        X_train = filtered_dataset.get_full_df()

        x_building = np.linspace(X_train['x'].min() - self.building_map_margin,
                                 X_train['x'].max() + self.building_map_margin,
                                 self.grid_size_in_each_dimension_in_each_building)

        y_building = np.linspace(X_train['y'].min() - self.building_map_margin,
                                 X_train['y'].max() + self.building_map_margin,
                                 self.grid_size_in_each_dimension_in_each_building)

        x_building, y_building = np.meshgrid(x_building, y_building)
        x_building, y_building = x_building.ravel(), y_building.ravel()

        floor_reproduction = np.zeros((len(x_building), 2))
        floor_reproduction[:, 0] = x_building
        floor_reproduction[:, 1] = y_building

        building_polygon = self.building_polygons[building]
        building_polygon_hole = self.building_polygons_holes[building]
        is_the_point_inside_the_building = verify_if_the_points_are_inside_the_polygon(floor_reproduction,
                                                                                       building_polygon)
        if building_polygon_hole is not None:
            is_the_point_inside_the_building_hole = verify_if_the_points_are_inside_the_polygon(floor_reproduction,
                                                                                       building_polygon_hole)
            is_the_point_inside_the_building = [is_the_point_inside_the_building[i] and 
                                                not is_the_point_inside_the_building_hole[i] 
                                                for i in range(len(is_the_point_inside_the_building))]
        x_building = np.array([x_building[i] for i in range(len(x_building)) if is_the_point_inside_the_building[i]])
        y_building = np.array([y_building[i] for i in range(len(y_building)) if is_the_point_inside_the_building[i]])

        return x_building, y_building

    def train(self, dataset):

        for building in range(self.num_buildings):

            x_building, y_building = self.construct_building_map(dataset, building)

            self.x_building[building] = x_building
            self.y_building[building] = y_building

            for floor in tqdm(range(self.num_floors_in_each_building[building])):

                filtered_dataset = dataset.get_floor_data(building=building, floor=floor)
                X_train = filtered_dataset.get_full_df()
                routers = self.get_all_routers_in_this_floor(X_train.columns)

                for router in routers:

                    if self.checking_non_null_minimum_percentage_of_samples(X_train, router):

                        self.power_probability_masks[building][floor][router] = {}

                        filtered_dataset = dataset.get_floor_data(building=building, floor=floor)
                        X_train = filtered_dataset.get_full_df()
                        X_train = self.get_mu_and_phi_estimation(X_train, router)

                        x_train = X_train['x'].to_numpy()
                        y_train = X_train['y'].to_numpy()
                        phi = X_train['phi'].to_numpy()
                        mu = X_train['mu'].to_numpy()

                        rbf_phi = Rbf(x_train, y_train, phi, function=self.radial_log_basis_function)
                        rbf_mu = Rbf(x_train, y_train, mu, function=self.radial_log_basis_function)

                        mu_building = rbf_mu(x_building, y_building)
                        phi_building = rbf_phi(x_building, y_building)

                        self.power_prior_probability_distribution[building][floor][router] = {}

                        total_num_samples_in_router = X_train[router].shape[0]

                        for power in range(0, self.max_power):
                            self.power_probability_masks[building][floor][router][
                                power] = self.approximate_position_density_function_given_router(power, mu_building,
                                                                                                 phi_building)

                            num_samples_with_value_power_in_router = (X_train[router] == power).sum()

                            self.power_prior_probability_distribution[building][floor][router][
                                power] = num_samples_with_value_power_in_router / total_num_samples_in_router

    def convert_power_db_to_base_used_in_the_model(self, power_db):
        if power_db == 100 or power_db is None or power_db == 0:
            return 0
        else:
            converted_power = power_db + 105
            return int(converted_power)

    def convert_base_used_in_the_model_to_power_db(self, power):
        if power == 0:
            return None
        else:
            power_db = power - 105
            return int(power_db)

    def get_valid_routers_building_and_floor(self, building, floor):
        return list(self.power_probability_masks[building][floor].keys())

    def plot_density_function_xy_given_bfrp(self, building, floor, router, power_db, image_save_path=None,
                                            figsize=(6, 4)):

        plt.figure(figsize=figsize)

        converted_power = self.convert_power_db_to_base_used_in_the_model(power_db)

        if converted_power == 0:
            plt.title(
                f'Estimated Probability Density Function of (x,y)\ngiven no power was received by the router {router}\nin the Floor {floor} of the Building {building}')
        else:
            plt.title(
                f'Estimated Probability Density Function of (x,y)\ngiven {int(power_db)}dB was received by the router {router}\nin the Floor {floor} of the Building {building}')

        plt.plot(self.building_polygons[building][:, 0], self.building_polygons[building][:, 1], c='black', linewidth=4)
        if self.building_polygons_holes[building] is not None:
            plt.plot(self.building_polygons_holes[building][:, 0], self.building_polygons_holes[building][:, 1], c='black', linewidth=4)
        plt.scatter(self.x_building[building], self.y_building[building],
                    c=self.power_probability_masks[building][floor][router][converted_power], s=9)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar()
        if image_save_path is not None:
            plt.savefig(image_save_path)
        plt.show()

    def plot_density_function_xy_given_bfr_for_different_p(self, building, floor, router, power_db_list, figsize=(6, 4),
                                                           image_save_path_list=None):

        if image_save_path_list is None:
            for power_db in power_db_list:
                self.plot_density_function_xy_given_bfrp(building, floor, router, power_db)
        else:
            for power_db, image_save_path in zip(power_db_list, image_save_path_list):
                self.plot_density_function_xy_given_bfrp(building, floor, router, power_db,
                                                         image_save_path=image_save_path)

    def get_registered_routers(self, building, floor):
        return self.power_probability_masks[building][floor].keys()

    def pred(self, X_test, selected_building=0, selected_floor=0):
        y_pred = []

        for index in X_test.index:

            test_sample = X_test.loc[index].to_frame().T

            distribution_xy_given_bf = {}
            normalization_term = {}

            count_count = {}

            for building in range(self.num_buildings):

                min_prob = self.eps * np.ones(len(self.x_building[building]))
                distribution_xy_given_bf[building] = {}
                normalization_term[building] = 0

                count_count[building] = {}

                for floor in range(self.num_floors_in_each_building[building]):

                    distribution_xy_given_bf[building][floor] = np.ones(len(self.x_building[building]))

                    count = 0

                    for router in self.get_valid_routers_building_and_floor(building, floor):

                        if router in test_sample.columns:
                            power = int(test_sample[router])

                            try:
                                prob_p_given_xybfr = self.power_probability_masks[building][floor][router][power]
                            except KeyError:
                                print(f"Error predicting building {building}, floor {floor}, router {router}, power{power}")
                                continue

                            count += 1

                            prob_p_given_xybfr = np.maximum(prob_p_given_xybfr, min_prob)
                            prob_p_given_xybfr = prob_p_given_xybfr / prob_p_given_xybfr.sum()
                            prob_xy_given_pbfr = prob_p_given_xybfr / (
                                    self.eps + self.power_prior_probability_distribution[building][floor][router][
                                power])

                            distribution_xy_given_bf[building][floor] = distribution_xy_given_bf[building][
                                                                            floor] * prob_xy_given_pbfr

                    # print(floor,count)
                    count_count[building][floor] = 4 ** count
                    distribution_xy_given_bf[building][floor] = (distribution_xy_given_bf[building][floor])
                    normalization_term[building] += distribution_xy_given_bf[building][floor].sum()
            
            # Deprecated categorical prediction
            """
            pred_building = 0
            prob_pred_building = 0
            prob_building = 0

            for building in range(self.num_buildings):
                prob_building = 0

                for floor in range(self.num_floors_in_each_building[building]):
                    # print(floor, np.log(distribution_xy_given_bf[building][floor].sum()) / np.log(4))
                    prob_building += distribution_xy_given_bf[building][floor].sum()

                if prob_building > prob_pred_building:
                    prob_pred_building = prob_building
                    pred_building = building

            pred_floor = 0
            prob_pred_floor = 0
            prob_floor = 0

            xpred_test = 0
            ypred_test = 0

            for floor in range(self.num_floors_in_each_building[pred_building]):

                distribution_xy_given_bf[pred_building][floor] = distribution_xy_given_bf[pred_building][floor] / \
                                                                 normalization_term[pred_building]

                xpred_test += sum(self.x_building[pred_building] * distribution_xy_given_bf[pred_building][floor])
                ypred_test += sum(self.y_building[pred_building] * distribution_xy_given_bf[pred_building][floor])

                prob_floor = distribution_xy_given_bf[pred_building][floor].sum()

                if prob_floor > prob_pred_floor:
                    prob_pred_floor = prob_floor
                    pred_floor = floor

            """

            pred_building = selected_building
            pred_floor = selected_floor

            xpred = sum(self.x_building[pred_building] * distribution_xy_given_bf[pred_building][pred_floor] /
                        distribution_xy_given_bf[pred_building][pred_floor].sum())
            ypred = sum(self.y_building[pred_building] * distribution_xy_given_bf[pred_building][pred_floor] /
                        distribution_xy_given_bf[pred_building][pred_floor].sum())

            # y_pred.append([pred_building, pred_floor, xpred, ypred, xpred_test, ypred_test])
            y_pred.append([pred_building, pred_floor, xpred, ypred])

        return np.array(y_pred), distribution_xy_given_bf
