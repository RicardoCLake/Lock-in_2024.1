import pickle

from localization.models.rbf_model import Model
from localization import dataset, utils
from os import path, mkdir

utils.make_deterministic(33)

num_buildings = 3
num_floors_in_each_building = {0: 4,
                               1: 4,
                               2: 5}
path_to_the_polygon_files_of_each_building = {0: 'data/geometry_building_0.npy',
                                              1: 'data/geometry_building_1.npy',
                                              2: 'data/geometry_building_2.npy'}

sigma = 6.75
num_samples_per_ap = 20

dataset = dataset.load_ujiindoor_loc(data_folder='data/filtered', transform=False)

model = Model(num_buildings,
              num_floors_in_each_building,
              path_to_the_polygon_files_of_each_building,
              sigma,
              num_samples_per_ap)
model.train(dataset)

if not path.exists('output'): mkdir('output')

with open('output/model.bin', 'wb') as outp:
    pickle.dump(model, outp, pickle.HIGHEST_PROTOCOL)
