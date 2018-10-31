import os

import numpy as np
import pandas as pd

import rampwf as rw
import json
from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types import BaseScoreType 

from PVChecker import *


class dummy_Predictions(BasePrediction):
  def __init__(self, y_pred=None, y_true=None, n_samples=None):
    if y_pred is not None:
      self.y_pred = y_pred
    elif y_true is not None:
      self.y_pred = y_true
    elif n_samples is not None:
      self.y_pred = np.empty(n_samples, dtype=object)
    else:
     raise ValueError('Missing init argument: y_pred, y_true, or n_samples')

  def __str__(self):
    return 'y_pred = {}'.format(self.y_pred)

  @classmethod

  #combination at the moment dummy implementation
  def combine(cls, predictions_list, index_list=None):
    if index_list is None:  # we combine the full list
      index_list = range(len(predictions_list))
    y_comb_list = [predictions_list[i].y_pred for i in index_list]

    n_preds = len(y_comb_list)
    y_preds_combined = np.empty(n_preds, dtype=object)
    #combined_predictions = cls(y_pred=predictions_list)
    combined_predictions = cls(y_pred=predictions_list[0].y_pred)
    return combined_predictions

  @property
  def valid_indexes(self):
    return self.y_pred != np.empty(len(self.y_pred), dtype=np.object)
    #return True


class dummy_score(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='dummy score', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        #we can us the python PVChecker -> need to transform data for it

        
        checker = PVChecker()
        #for MC_PVs in y_true_label_index:
        #loop over event
        for i_event in range(0, len(y_true_label_index)):
          MCPV_arr_tot = []
          RecPV_arr_tot = []
          #set-up MC PVs
          for MC_PV in y_true_label_index[i_event]:
            MCPV_arr = np.array([MC_PV.x, MC_PV.y, MC_PV.z, MC_PV.numberTracks])
            MCPV_arr_tot = MCPV_arr_tot + [MCPV_arr]
          #set-up reconstructed PVs
          for Rec_PV in y_pred_label_index[i_event]:
            RecPV_arr = np.array(Rec_PV)
            RecPV_arr_tot =  RecPV_arr_tot + [RecPV_arr]
          MCPV_arr_tot = np.array(MCPV_arr_tot)
          RecPV_arr_tot = np.array(RecPV_arr_tot)
          checker.load_data(RecPV_arr_tot, MCPV_arr_tot)
          checker.check_event_df()

        #checker = PVChecker
        checker.calculate_eff()
        
        return checker.reconstructible_efficiency

    def check_y_pred_dimensions(self, y_true, y_pred):
      if len(y_true) != len(y_pred):
        raise ValueError('sWrong y_pred dimensions: y_pred should have {} instances, ''instead it has {} instances'.format(len(y_true), len(y_pred)))




problem_title = 'Mars craters detection and classification'
# A type (class) which will be used to create wrapper objects for y_pred
dummypd = dummy_Predictions
Predictions = dummypd
# An object implementing the workflow
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
    #first entry in tuple is for training, the second for testing
    #number of tuples gives number of crossfolds
    return [
            (np.r_[0:n_tot], np.r_[0:n_tot])]


class MCVertex:
  def __init__(self, x, y, z, numberTracks):
    self.x = x
    self.y = y
    self.z = z
    self.numberTracks = numberTracks
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
      mc_pvs = [ MCVertex(*h['Pos'], + h['products'] ) for key,h in MCVertices.items() ]
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
    return x_array, y_array
    #return np.array([[(1,2)],[(1,2)]]),np.array([[(2,3)],[(1,2)]])


def get_test_data(path='/home/freiss/lxplus_work/public/recept/RAPID-data3/'):
    return _read_data(path, 'test')
    #return np.array([1]),np.array([2])


def get_train_data(path='/home/freiss/lxplus_work/public/recept/RAPID-data3/'):
    return _read_data(path, 'train')




