from rampwf.prediction_types.base import BasePrediction
import numpy as np

class PVPredictions(BasePrediction):
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