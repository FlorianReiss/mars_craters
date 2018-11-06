import os

import numpy as np
import pandas as pd

import rampwf as rw
import json
from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types import BaseScoreType 

# our custom implementations for predictions and scoring
from PVPredictions import PVPredictions
from PVScore import PVScore, PVScore_total

# ramp-kit for the RAPID challenge
# test and train data should be in data/test and data/train
# one has to implement the ObjectDetector class in submissions/<yoursubmission>/object_detector.py
# with the functions fit(X,y) and predict(X), where X is the input data and y the truth information
# the fit function contains the training of the model, while the predict function applies it


#to do: implement various scoring algorithms
#to do: decide on crossfolds, prediction




problem_title = 'RAPID challenge'
# A type (class) which will be used to create wrapper objects for y_pred

Predictions = PVPredictions
# An object implementing the workflow
workflow = rw.workflows.ObjectDetector()

# The overlap between adjacent patches is 56 pixels
# The scoring region is chosen so that despite the overlap,
# no crater is scored twice, hence the boundaries of
# 28 = 56 / 2 and 196 = 224 - 56 / 2



score_types = [
    PVScore(), PVScore_total(name = "efficiency",mode="eff"), PVScore_total(name = "fake rate",mode="fake"), PVScore_total(name = "total",mode="total")
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


#helper calss to hold Velo state + covariance matrix
class VeloState:
  def __init__(self, x, y, z, tx, ty, pq, cov_x, cov_y, cov_tx, cov_ty, cov_xtx): 
    self.x = x
    self.y = y
    self.z = z
    self.tx = tx
    self.ty = ty
    self.pq = pq
    self.cov_x = cov_x
    self.cov_y = cov_y
    self.cov_tx = cov_tx
    self.cov_ty = cov_ty
    self.cov_xtx = cov_xtx
  def __repr__(self):
      return 'VeloState({0},{1},{2},{3},{4})'.format(self.x,self.y,self.z,self.tx,self.ty)
  def __str__(self):
      return 'VeloState({0},{1},{2},{3},{4})'.format(self.x,self.y,self.z,self.tx,self.ty)

# helper class to hold Velo hits
class VeloHit:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __repr__(self):
      return 'VeloHit({0}, {1}, {2})'.format(self.x,self.y,self.z)
  def __str__(self):
      return 'VeloHit({0}, {1}, {2})'.format(self.x,self.y,self.z)



#class to hold tracks and hits of an event
class EventData:
  def __init__(self, list_tracks, list_hits):
    self.tracks = list_tracks
    self.hits = list_hits

  def __repr__(self):
    return 'EventData'

  def __str__(self):
      return 'EventData'




def _read_data(path, type):
    """
    Read and process data and labels.

    Parameters
    ----------
    path : path to directory that has 'data' subdir
    typ : {'train', 'test'}

    Returns
    -------
    X, y data
    X: np array of EventData, where EventData conists of list of VeloStates and list of VeloHits for a event

    Y: np array of lists of MCVertex, where a list contains all MC vertices of an event, and the array contains the lists of all events
    """
    #loop over all json files:

    list_y = []
    list_x = []
    #default path is .
    #have to set it for reading
    path = path + '/data/{0}/'.format(type)
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
      VeloHits  = jdata['VPClusters']
      #velo_states = [tuple(h['ClosestToBeam']) for key,h in VeloTracks.items() ]
      velo_states = [VeloState(*h['ClosestToBeam'], *h['errCTBState'])  for key,h in VeloTracks.items() ]
      velo_hits = [VeloHit(h['x'], h['y'], h['z']) for key, h in VeloHits.items()]
      event = EventData(velo_states, velo_hits)
      #velo_states = [VeloState(*h['ClosestToBeam'])  for key,h in VeloTracks.items() ]
      #velo_states_cov = [VeloState_Cov(*h['errCTBState']) for key,h in VeloTracks.items() ]

      #check if this indeed puts correct velo state and cov matrix together
      #zipped_x = [i for i in zip(velo_states, velo_states_cov)]

     # list_x = list_x + [zipped_x]
      list_x = list_x + [event]
      

    y_array = np.empty(len(list_y), dtype=object)
    y_array[:] = list_y

    x_array = np.empty(len(list_x), dtype=object)
    x_array[:] = list_x
    x_array = np.array(x_array)
    print("Rec data")
    #print(x_array)
    #print(event)
    #print(event.tracks)
    #print(event.hits)

    return x_array, y_array
    #return np.array([[(1,2)],[(1,2)]]),np.array([[(2,3)],[(1,2)]])


def get_test_data(path):
    return _read_data(path, 'test')
    #return np.array([1]),np.array([2])


def get_train_data(path):
    return _read_data(path, 'train')




