import os
import json
import numpy as np
from pprint import pprint
import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline 
plt.style.use('ggplot')






def z_dist_matched(rec_pv_z, mc_pv_z, m_distance):
  return abs(rec_pv_z - mc_pv_z) < m_distance

#define matching criteria, based on distance:
def is_matched(rec_pv_x=0., rec_pv_y=0., rec_pv_z=0., mc_pv_x=0., mc_pv_y=0., mc_pv_z=0., m_distance=0.1):
  return z_dist_matched(rec_pv_z, mc_pv_z, m_distance)
#define configuration
m_distance = 0.5

#class to do checking and plots
class PVChecker:
  def __init__(self):
    
    #configuration for matching
    self.m_mintracks = 1
    self.m_distance = 0.3

    #data frames to collect all rec and true pvs from all events
    self.df_all_events_true_rec_pvs = pd.DataFrame()
    self.df_all_events_fake_rec_pvs = pd.DataFrame()
    self.df_all_events_mc_pvs = pd.DataFrame()


  #load data
  #here we expect arr_rec_pvs to be numpy array of array[x,y,z] and arr_mc_pvs to be numpy array of array[x,y,z, nTracks]
  def load_data(self, arr_rec_pvs, arr_mc_pvs):
    self.df_rec_pvs = pd.DataFrame(arr_rec_pvs)
    self.df_rec_pvs['matched'] = 0
    #entry number of matched MC PV, -99 if not matched
    self.df_rec_pvs['matched_pv_key'] = -99
    self.df_rec_pvs.columns=['x', 'y', 'z','matched', 'matched_pv_key']
    self.df_mc_pvs = pd.DataFrame(arr_mc_pvs)
    self.df_mc_pvs.columns=['x', 'y', 'z','nVeloTracks']

  

    #check event with previously loaded data frames
  def check_event_df(self):
    #loop over MC PVs and find rec PV with minimum z distance
    for mc_index, mc_pv in self.df_mc_pvs.iterrows():
      #if mc_pv['nVeloTracks'] < self.m_mintracks: continue
      
      #loop over rec PVs
      true_z = mc_pv['z']
      min_dist = 10000.
      index_min_dist = -99
      matched_pv_key = -99
      for rec_index, rec_pv in self.df_rec_pvs.iterrows():
        rec_z = rec_pv['z']
        dist_z = abs(true_z - rec_z)
        if dist_z < min_dist:
          min_dist = dist_z
          index_min_dist = rec_index
      rec_z = self.df_rec_pvs['z'][index_min_dist]
      dist_z =  abs(true_z - rec_z)
      #match rec and MC PVs, if the rec pv with minimum z distance to MC PV fullfills matching crtierion
      if is_matched(rec_pv_z = rec_z, mc_pv_z=true_z, m_distance=self.m_distance) and not self.df_rec_pvs['matched'][index_min_dist]:
        self.df_rec_pvs.loc[index_min_dist, 'matched'] = 1
        self.df_rec_pvs.loc[index_min_dist, 'matched_pv_key'] = mc_index
        


    #test creating sub dataframes of real and fake rec pv
    df = self.df_rec_pvs[self.df_rec_pvs.matched == 1]
    self.df_fake_rec_pvs = self.df_rec_pvs[self.df_rec_pvs.matched == 0]

    df_true = pd.DataFrame(columns=['true_x', 'true_y', 'true_z'], dtype=float)
    for key,row in df.iterrows():
      df_true.loc[key,['true_x','true_y','true_z']] = self.df_mc_pvs.loc[row['matched_pv_key'],['x','y','z']].values
    df = pd.concat([df,df_true], axis = 1)
    for dim in ['x', 'y', 'z']:
      df['residual_'+dim] = df[dim] - df['true_'+dim] 
    self.df_true_rec_pvs = df

    self.df_all_events_true_rec_pvs = self.df_all_events_true_rec_pvs.append(self.df_true_rec_pvs, ignore_index=True)
    self.df_all_events_fake_rec_pvs = self.df_all_events_fake_rec_pvs.append(self.df_fake_rec_pvs, ignore_index=True)
    self.df_all_events_mc_pvs       = self.df_all_events_mc_pvs.append(self.df_mc_pvs, ignore_index=True)

  #print efficiencies and fake rate
  def print_eff(self):
    #use total data frames to count found/total PVs
    counter_found_MC_PV = self.df_all_events_true_rec_pvs.index.size
    counter_total_MC_PV = self.df_all_events_mc_pvs.index.size
    counter_total_MC_PV_reconstructible = self.df_all_events_mc_pvs[self.df_all_events_mc_pvs.nVeloTracks > self.m_mintracks].index.size
    #counter_total_MC_PV = self.df_all_events_mc_pvs.index.size
    counter_fake_PV = self.df_all_events_fake_rec_pvs.index.size

    self.total_efficiency = counter_found_MC_PV/counter_total_MC_PV
    self.total_fake_rate = counter_fake_PV/(counter_found_MC_PV + counter_fake_PV)
    self.reconstructible_efficiency = counter_found_MC_PV/counter_total_MC_PV_reconstructible
    print ("found", counter_found_MC_PV, "of", counter_total_MC_PV, "primary vertices") 
    print ("efficiency:", self.total_efficiency)
    print (counter_total_MC_PV_reconstructible,"of", counter_total_MC_PV, "PVs are reconstructible (have more than", self.m_mintracks, "reconstructed Velo tracks)")
    print ("reconstructible PV efficiency: ", self.reconstructible_efficiency)
    print ("have", counter_fake_PV, "fake PVs")
    print ("fake rate:", self.total_fake_rate)
  #function to get determine total score
  def final_score(self):
    #critertia: efficiency, fake rate, sigma of residuals, means of residuals?
    fin_score = self.reconstructible_efficiency + self.total_fake_rate
    self.fin_score = fin_score / 2.
    print("the final score is", self.fin_score, "!")


