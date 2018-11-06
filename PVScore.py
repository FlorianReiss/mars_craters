import numpy as np
from rampwf.score_types import BaseScoreType 
from PVChecker import PVChecker


class PVScore(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='PV rec eff', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        #we can us the python PVChecker -> need to transform data for it

        
        checker = PVChecker()
        checker.load_from_ramp(y_true_label_index, y_pred_label_index)
        

        #checker = PVChecker
        checker.calculate_eff()
        
        return checker.reconstructible_efficiency

    def check_y_pred_dimensions(self, y_true, y_pred):
      if len(y_true) != len(y_pred):
        raise ValueError('Wrong y_pred dimensions: y_pred should have {} instances, ''instead it has {} instances'.format(len(y_true), len(y_pred)))


class PVScore_total(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, mode, name='total score', precision=2):
        self.name = name
        self.precision = precision
        self.mode = mode


    def __call__(self, y_true_label_index, y_pred_label_index):
        #we can us the python PVChecker -> need to transform data for it

        
        checker = PVChecker()
        checker.load_from_ramp(y_true_label_index, y_pred_label_index)
        

        #checker = PVChecker
        checker.calculate_eff()
        checker.final_score()
        if self.mode == "total":
          return checker.fin_score
        if self.mode == "eff":
          return checker.reconstructible_efficiency
        if self.mode == "fake":
          return checker.total_fake_rate

    def check_y_pred_dimensions(self, y_true, y_pred):
      if len(y_true) != len(y_pred):
        raise ValueError('Wrong y_pred dimensions: y_pred should have {} instances, ''instead it has {} instances'.format(len(y_true), len(y_pred)))
