from scipy.special import gamma
from scipy.special import hyp2f1
import pandas as pd
import numpy as np
import scipy

class TLoc:         

    def __init__(self, train_data: pd.DataFrame, pred_class="room"):
        # self.non_null_minimum_percentage = 0.1
        self.train_data = train_data
        # self.aps = self.get_aps_with_non_zero_minimum_percentage(self.train_data)
        cols = self.train_data.columns
        self.aps = list(cols[cols != pred_class])
        if len(self.aps) == 1:
            self.max_power = int(self.train_data[self.aps].max())
        else:
            self.max_power = int(self.train_data[self.aps].max().max())

        self.pred_class = pred_class
        self.spaces = list(self.train_data[self.pred_class].unique())

        self.power_probability_masks = {}
        self.power_prior_probability_distribution = {}
        self.eps = 1e-5


    def get_aps_with_non_zero_minimum_percentage(self, data):
        percentage_of_non_zeros = (data.iloc[:, 5:] != 0).sum() / data.shape[0]
        return data.columns[5:][percentage_of_non_zeros >= self.non_null_minimum_percentage]
    


    def get_mu_and_phi_estimation(self, data, router):
        mu = []
        phi = []
        data_of_router = data[[self.pred_class, router]]
        for space in self.spaces:
            data_of_router_in_space = data_of_router[data_of_router[self.pred_class] == space]

            data_of_router_in_space_without_zero_values = data_of_router_in_space[data_of_router_in_space[router] != 0]
            if len(data_of_router_in_space_without_zero_values) == 0:
                mu.append(0.0)
            else:
                mu.append(data_of_router_in_space_without_zero_values[router].mean())
            phi.append(1 - data_of_router_in_space_without_zero_values.shape[0] / data_of_router_in_space.shape[0])

        return mu, phi



    def train(self):
        for router in self.aps:


            self.power_probability_masks[router] = {}
            self.power_prior_probability_distribution[router] = {}

            mu, phi = self.get_mu_and_phi_estimation(self.train_data, router)


            total_num_samples_in_router = self.train_data[router].shape[0]
            for power in range(0, self.max_power):
                self.power_probability_masks[router][
                    power] = self.approximate_position_density_function_given_router(power, np.array(mu),
                                                                                    np.array(phi))
                num_samples_with_value_power_in_router = (self.train_data[router] == power).sum()

                self.power_prior_probability_distribution[router][
                    power] = num_samples_with_value_power_in_router / total_num_samples_in_router

                    

    def cumulative_distribution_function_of_t_student(self, x, v):

        return 0.5 + x * gamma((v + 1) / 2) * hyp2f1(1 / 2, (v + 1) / 2, 3 / 2, -(x ** 2) / v) / (
                np.sqrt(v * np.pi) * gamma(v / 2))
    
    def cumulative_distribution_function_of_power(self, power, mu, phi, sigma, v):

        cdf = phi * np.heaviside(power, 1) + (1 - phi) * self.cumulative_distribution_function_of_t_student(
            (power - mu) / sigma, v)
        
        return cdf
    

    def approximate_position_density_function_given_router(self, power, mu, phi, sigma=5, num_samples_per_ap=30, t_score_alpha=0.05):

        v = np.ceil(num_samples_per_ap * (1 - phi) - 1)
        v = np.where(v <= 0, 1, v)

        t_score = scipy.stats.t.ppf(0.5 + t_score_alpha, v)

        density_function = self.cumulative_distribution_function_of_power(power + t_score * sigma, mu, phi, sigma, v) - self.cumulative_distribution_function_of_power(
                    power - t_score * sigma, mu, phi, sigma, v)  # power, mu, phi, sigma, v

        return density_function
    

    def pred(self, X_test):

        y_pred = []
        min_prob = self.eps * np.ones(len(self.spaces))

        for _, test_sample in X_test.iterrows():

            distribution_xy_given_bf = np.ones(len(self.spaces))

            for router in self.aps:

                    power = int(test_sample[router])

                    try:
                        prob_p_given_xybfr = self.power_probability_masks[router][power]
                    except KeyError:
                        # print(f"Error predicting router {router}, power {power}")
                        continue
                    


                    prob_p_given_xybfr = np.maximum(prob_p_given_xybfr, min_prob)
                    #prob_p_given_xybfr = prob_p_given_xybfr / prob_p_given_xybfr.sum()
                    #prob_xy_given_pbfr = prob_p_given_xybfr / (
                    #                tloc.eps + tloc.power_prior_probability_distribution[router][power])


                    distribution_xy_given_bf = distribution_xy_given_bf * prob_p_given_xybfr




            room_pred = self.spaces[distribution_xy_given_bf.argmax()]
            y_pred.append(room_pred)

        y_pred = np.array(y_pred)
        ground_truth = np.array(list(X_test[self.pred_class]))
        ac = np.sum(y_pred == ground_truth)/len(ground_truth)
        return ac, y_pred, ground_truth





