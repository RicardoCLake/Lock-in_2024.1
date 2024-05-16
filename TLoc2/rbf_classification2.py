from scipy.special import gamma
from scipy.special import hyp2f1
import pandas as pd
import numpy as np
import scipy
from sklearn.mixture import GaussianMixture

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
        self.classes = list(self.train_data[self.pred_class].unique())
        self.power_probability_masks = {}
        self.power_prior_probability_distribution = {}
        self.eps = 1e-5

    def get_aps_with_non_zero_minimum_percentage(self, data):
        percentage_of_non_zeros = (data.iloc[:, 5:] != 0).sum() / data.shape[0]
        return data.columns[5:][percentage_of_non_zeros >= self.non_null_minimum_percentage]
    
    def get_parameters_estimation(self, data, router):
        means = []
        weights = []
        phi = []
        data_of_router = data[[self.pred_class, router]]
        for class_ in self.classes:
            data_of_router_in_class = data_of_router[data_of_router[self.pred_class] == class_]

            data_of_router_in_class_without_zero_values = data_of_router_in_class[data_of_router_in_class[router] != 0]
            if len(data_of_router_in_class_without_zero_values) == 0:
                means.append([0.0,0.0])
                weights.append([1.0,0.0])
            elif len(data_of_router_in_class_without_zero_values) == 1:
                means.append([data_of_router_in_class_without_zero_values[router],0.0])
                weights.append([1.0,0.0])
            else:
                gmm = GaussianMixture(n_components=2, covariance_type='tied')
                gmm.fit(np.array(data_of_router_in_class_without_zero_values[router]).reshape(-1,1))
                means.append([gmm.means_[0][0],gmm.means_[1][0]])
                weights.append(gmm.weights_)
                
            phi.append(1 - data_of_router_in_class_without_zero_values.shape[0] / data_of_router_in_class.shape[0])
        return means, weights, phi

    def train(self): #### preciso arrumar ainda essa função
        for router in self.aps:

            self.power_probability_masks[router] = {}
            self.power_prior_probability_distribution[router] = {}

            means, weights, phi = self.get_parameters_estimation(self.train_data, router)

            total_num_samples_in_router = self.train_data[router].shape[0]
            for power in range(0, self.max_power):
                self.power_probability_masks[router][power] = self.approximate_position_density_function_given_router(power, np.array(means), np.array(weights), np.array(phi))
                num_samples_with_value_power_in_router = (self.train_data[router] == power).sum()

                self.power_prior_probability_distribution[router][
                    power] = num_samples_with_value_power_in_router / total_num_samples_in_router

    def cumulative_distribution_function_of_gaussian_mixture(self, x, means, sigma, weights):
        tmp1 = weights[:,0] 
        tmp2 = scipy.stats.norm.cdf(x, means[:,0], sigma)
        return weights[:,0]*scipy.stats.norm.cdf(x, means[:,0], sigma) + weights[:,1]*scipy.stats.norm.cdf(x, means[:,1], sigma)
    
    def cumulative_distribution_function_of_power(self, power, means, weights, phi, sigma):
        
        cdf = phi * np.heaviside(power, 1) + (1 - phi) * self.cumulative_distribution_function_of_gaussian_mixture(power, means, sigma, weights)
        return cdf
    
    def approximate_position_density_function_given_router(self, power, means, weights, phi, sigma=5, num_samples_per_ap=30, score_alpha=0.05):

        score = scipy.stats.norm.ppf(0.5 + score_alpha)          
        density_function = self.cumulative_distribution_function_of_power(power + score * sigma, means, weights, phi, sigma) - self.cumulative_distribution_function_of_power(
                    power - score * sigma, means, weights, phi, sigma)  

        return density_function
    
    def pred(self, X_test):

        y_pred = []
        min_prob = self.eps * np.ones(len(self.classes))

        for _, test_sample in X_test.iterrows():

            distribution_xy_given_bf = np.ones(len(self.classes))

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

            room_pred = self.classes[distribution_xy_given_bf.argmax()]
            y_pred.append(room_pred)

        y_pred = np.array(y_pred)
        ground_truth = np.array(list(X_test[self.pred_class]))
        ac = np.sum(y_pred == ground_truth)/len(ground_truth)
        return ac, y_pred, ground_truth





