# -*- coding: utf-8 -*-
"""
the minimum script to run 10 seconds of black screen
"""

import matplotlib.pyplot as plt
import WarpedVisualStim.StimulusRoutines as stim
from WarpedVisualStim.MonitorSetup import Monitor, Indicator
from WarpedVisualStim.DisplayStimulus import DisplaySequence

# Initialize Monitor object
mon_resolution = (1440, 2560)  # enter your monitors resolution
mon_width_cm = 60.  # enter your monitors width in cm
mon_height_cm = 34.  # enter your monitors height in cm
mon_refresh_rate = 60  # enter your monitors height in Hz
mon_C2T_cm = mon_height_cm / 2.		
mon_C2A_cm = mon_width_cm / 2.		
mon_center_coordinates = (0., 60.) 	#these should be the angular coordinates of the center of the screen with respect to mouse
mon_dis = 15.
mon_downsample_rate = 1 		#what is this?

mon = Monitor(resolution=mon_resolution, dis=mon_dis, mon_width_cm=mon_width_cm,
              mon_height_cm=mon_height_cm, C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm,
              center_coordinates=mon_center_coordinates,
              downsample_rate=mon_downsample_rate)



ind_width_cm = 1.
ind_height_cm = 1.
ind_position = 'southeast'
ind_is_sync = True
ind_freq = 1 		# or 1?? what difference does it make?


ind = Indicator(mon, width_cm=ind_width_cm, height_cm=ind_height_cm,
                position=ind_position, is_sync=ind_is_sync, freq=ind_freq)


ds_log_dir = r'D:\data\displaySequence'

ds = DisplaySequence(log_dir=ds_log_dir, is_by_index=True)
dgc = stim.DriftingGratingCircle(monitor=mon, indicator=ind, background=0.,
                                 coordinate='degree', center=(0., 20.), sf_list=(0.02,),
                                 tf_list=(2.,), dire_list=(180.,), con_list=(0.8,), radius_list=(20.,),
                                 block_dur=1., midgap_dur=1., iteration=1, pregap_dur=1.,
                                 postgap_dur=1., is_blank_block=False, is_random_start_phase=False)
ds.set_stim(dgc)
ds.trigger_display()
plt.show()