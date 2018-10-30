import os

import numpy as np
import pandas as pd

import rampwf as rw
import json



class dummy_score():
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='dummy score', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        score = 0.5
        return score
    def score_function(self, ground_truths, predictions, valid_indexes=[0]):
        print ("ground truths")
        print(ground_truths)
        print("predictions")
        print(predictions)
        return self(ground_truths, predictions)



problem_title = 'Mars craters detection and classification'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_detection()
# An object implementing the workflow
print ('this works?')
workflow = rw.workflows.ObjectDetector()

# The overlap between adjacent patches is 56 pixels
# The scoring region is chosen so that despite the overlap,
# no crater is scored twice, hence the boundaries of
# 28 = 56 / 2 and 196 = 224 - 56 / 2
minipatch = [28, 196, 28, 196]

score_types = [
   
    dummy_score()
]


def get_cv(X, y):
    # 3 quadrangles for training have not exactly the same size,
    # but for simplicity just cut in 3
    # for each fold use one quadrangle as test set, the other two as training

    n_tot = len(X)
    n1 = n_tot // 3
    n2 = n1 * 2

    return [
            (np.r_[n1:n_tot], np.r_[0:n1]),
            (np.r_[0:n1, n2:n_tot], np.r_[n1:n2])]


class MCVertex:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
  def __repr__(self):
      #return '{0}, {1}, {2}'.format(self.x, self.y, self.z)
      return 'MCVertex'
  def __str__(self):
        return 'MCVertex'


class VeloState:
  def __init__(self, x, y, z, tx, ty, pq): 
    self.x = x
    self.y = y
    self.z = z
    self.tx = tx
    self.ty = ty
  def __repr__(self):
      return 'VeloState'
  def __str__(self):
        return 'VeloState'



class VeloState_Cov:
  def __init__(self, cov_x, cov_y, cov_tx, cov_ty, cov_xtx):
    self.cov_x = cov_x
    self.cov_y = cov_y
    self.cov_tx = cov_tx
    self.cov_ty = cov_ty
    self.cov_xtx = cov_xtx
  def __repr__(self):
      return 'Cov'
  def __str__(self):
        return 'Cov'


def _read_data(path, bla='bla'):
    """
    Read and process data and labels.

    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'train', 'test'}

    Returns
    -------
    X, y data

    """
    #loop over all json files:

    list_y = []
    list_x = []
    #default path is .
    #have to set it for reading
    path = path + '/data/'
    for file in os.listdir(path):
      if not file.endswith('.json'): continue
      file_path = path + file
      jdata = json.load(open(file_path))
      MCVertices  = jdata['MCVertices']
      #mc_pvs = np.array([ np.array(h['Pos'] + [h['products']] ) for key,h in MCVertices.items() ])
      #mc_pvs = [ tuple(h['Pos'] + [h['products']] ) for key,h in MCVertices.items() ]
      mc_pvs = [ MCVertex(*h['Pos'] ) for key,h in MCVertices.items() ]
      list_y = list_y + [mc_pvs]

      VeloTracks  = jdata['VeloTracks']
      #velo_states = [tuple(h['ClosestToBeam']) for key,h in VeloTracks.items() ]
      velo_states = [VeloState(*h['ClosestToBeam']) for key,h in VeloTracks.items() ]
      velo_states_cov = [VeloState_Cov(*h['errCTBState']) for key,h in VeloTracks.items() ]

      #check if this indeed puts correct velo state and cov matrix together
      zipped_x = [i for i in zip(velo_states, velo_states_cov)]

      list_x = list_x + [zipped_x]

    y_array = np.empty(len(list_y), dtype=object)
    y_array[:] = list_y

    x_array = np.empty(len(list_x), dtype=object)
    x_array[:] = list_x
    x_array = np.array(x_array)
    print("print x array")
    print(x_array)
    print((type(x_array)))
    print((type(x_array[0])))
    return x_array, y_array
    #return np.array([[(1,2)],[(1,2)]]),np.array([[(2,3)],[(1,2)]])


def get_test_data(path='/home/freiss/lxplus_work/public/recept/RAPID-data3/'):
    return _read_data(path, 'test')
    #return np.array([1]),np.array([2])


def get_train_data(path='/home/freiss/lxplus_work/public/recept/RAPID-data3/'):
    return _read_data(path, 'train')




