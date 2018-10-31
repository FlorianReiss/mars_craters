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
        raise ValueError('Wrong y_pred dimensions: y_pred should have {} instances, ''instead it has {} instances'.format(len(y_true), len(y_pred)))

