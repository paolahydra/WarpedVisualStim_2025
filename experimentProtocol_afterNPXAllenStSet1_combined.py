import os
import numpy as np
import matplotlib.pyplot as plt
import WarpedVisualStim as rm
import WarpedVisualStim.StimulusRoutines as stim
from WarpedVisualStim.DisplayStimulus import DisplaySequence
from WarpedVisualStim.MonitorSetup import Monitor, Indicator


# # Initialize Monitor object
# ds_psychopy_mon = 'testMonitor'
# ds_display_screen = 1               # indicate here the identifier of the monitor where to present stimuli
# mon_resolution = (1440, 2560)       # enter your monitors resolution
# mon_width_cm = 60.                  # enter your monitors width in cm
# mon_height_cm = 34.                 # enter your monitors height in cm
# mon_refresh_rate = 60               # enter your monitors height in Hz
# mon_C2T_cm = mon_height_cm / 2.		
# mon_C2A_cm = mon_width_cm / 2.		
# mon_center_coordinates = (0., 60.) 	# these should be the angular coordinates of the center of the screen with respect to mouse
# mon_dis = 15.
# mon_downsample_rate = 5 		    # no downsampling slows down stimulus presentation, 5 was tested with previous monitor


# Initialize Monitor object
ds_psychopy_mon = 'ASUS_PA248Q'
ds_display_screen = 1               # indicate here the identifier of the monitor where to present stimuli
mon_resolution = (1200, 1920)       # enter your monitors resolution
mon_width_cm = 55.7                 # enter your monitors width in cm
mon_height_cm = 34.8                # enter your monitors height in cm
mon_refresh_rate = 60               # enter your monitors height in Hz
mon_C2T_cm = mon_height_cm / 2.		
mon_C2A_cm = mon_width_cm / 2.		
mon_center_coordinates = (0., 60.) 	# these should be the angular coordinates of the center of the screen with respect to mouse
mon_dis = 15.
mon_downsample_rate = 4 		    # no downsampling slows down stimulus presentation, 4 should be a good target for ASUS_PA248Q



# General data settings
ds_log_dir = r'C:\data\visual_display_log'
ds_backupdir = None
ds_identifier = 'BrainObservatory1'
ds_display_iter = 1
ds_mouse_id = 'MOUSE'
ds_user_id = 'USER'

ds_is_by_index = True
ds_is_interpolate = False
ds_is_triggered = False
ds_is_save_sequence = True
ds_trigger_event = "negative_edge"
ds_trigger_NI_dev = 'Dev1'
ds_trigger_NI_port = 1
ds_trigger_NI_line = 0
ds_is_sync_pulse = False
ds_sync_pulse_NI_dev = 'Dev1'
ds_sync_pulse_NI_port = 1
ds_sync_pulse_NI_line = 1
ds_initial_background_color = 0.
ds_color_weights = (1., 1., 1.)


# Indicator settings
ind_width_cm = 1.
ind_height_cm = 1.
ind_position = 'southeast'
ind_is_sync = True
ind_freq = 1 


# ========= Initialize the Monitor, Indicator and DisplaySequence objects ===================
mon = Monitor(resolution=mon_resolution, dis=mon_dis, mon_width_cm=mon_width_cm,
              mon_height_cm=mon_height_cm, C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm,
              center_coordinates=mon_center_coordinates,
              downsample_rate=mon_downsample_rate)


ind = Indicator(mon, width_cm=ind_width_cm, height_cm=ind_height_cm,
                position=ind_position, is_sync=ind_is_sync, freq=ind_freq)


ds = DisplaySequence(log_dir=ds_log_dir, backupdir=ds_backupdir,
                     identifier=ds_identifier, display_iter=ds_display_iter,
                     mouse_id=ds_mouse_id, user_id=ds_user_id,
                     psychopy_mon=ds_psychopy_mon, is_by_index=ds_is_by_index,
                     is_interpolate=ds_is_interpolate, is_triggered=ds_is_triggered,
                     trigger_event=ds_trigger_event, trigger_NI_dev=ds_trigger_NI_dev,
                     trigger_NI_port=ds_trigger_NI_port, trigger_NI_line=ds_trigger_NI_line,
                     is_sync_pulse=ds_is_sync_pulse, sync_pulse_NI_dev=ds_sync_pulse_NI_dev,
                     sync_pulse_NI_port=ds_sync_pulse_NI_port,
                     sync_pulse_NI_line=ds_sync_pulse_NI_line,
                     display_screen=ds_display_screen, is_save_sequence=ds_is_save_sequence,
                     initial_background_color=ds_initial_background_color,
                     color_weights=ds_color_weights)


# building stimuli after brain observatory 1.1 - Neuropixels visual coding stimulus set 1 (with some changes):
# https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/80/75/8075a100-ca64-429a-b39a-569121b612b2/neuropixels_visual_coding_-_white_paper_v10.pdf
# see also:
# https://cdck-file-uploads-canada1.s3.dualstack.ca-central-1.amazonaws.com/flex027/uploads/brainobservatory/original/2X/d/d7eeb93a17be13d8a7181c726b8f06a47f74795d.pdf
# https://observatory.brain-map.org/visualcoding/stimulus/drifting_gratings
# https://observatory.brain-map.org/visualcoding/stimulus/static_gratings
# https://observatory.brain-map.org/visualcoding/stimulus/natural_scenes

# ================ Receptive field mapping - drifting gratings gabor ================ 

# input center parameters
# x_range = (5.0, 115.0)    # [xmin, xmax]
# y_range = (25.0, -25.0)   # preserves given orientation (top→bottom)
# n_x = 11                  # how much you want to tile x axis
# n_y = 5                   # how much you want to tile y axis
# #alternative 1:
x_range = (7.5, 112.5)   # [xmin, xmax]
y_range = (37.5, -37.5)   # preserves given orientation (top→bottom)
n_x = 7                   # how much you want to tile x axis
n_y = 5                   # how much you want to tile y axis
# # #alternative 2:
# x_range = (-7.5, 127.5)   # [xmin, xmax]
# y_range = (52.5, -52.5)   # preserves given orientation (top→bottom)
# n_x = 9                   # how much you want to tile x axis
# n_y = 7                   # how much you want to tile y axis
# calculate grid of centers
x_edges = np.linspace(x_range[0], x_range[1], n_x + 1)
y_edges = np.linspace(y_range[0], y_range[1], n_y + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
print(x_centers)
print(y_centers)
# n x n grid of centers, then flatten to list of tuples (x,y)
Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='xy')  # shape (n_x, n_y)


rf_center_list = list(map(tuple, np.column_stack([Xc.ravel(), Yc.ravel()])))


# directions used are not mentioned anywhere... nor how to handle it in analysis...
# rf_direlist = (0.,90.,180.,270., 45.,135.,225.,315.,)
# rf_iterationN = 10
rf_direlist = (0.,90.,180.,270.,)
rf_iterationN = 20                  
rf_block_dur = 0.250
rf_midgapdur = 0.


# #test:
# rf_center_list = [(10., 20.),(-10., 100.)]
# rf_direlist = (0.,90.,)
# rf_iterationN = 2
# rf_block_dur = 1.
# rf_midgapdur = 0.

rf = stim.DriftingGratingMultipleCircle(monitor=mon, indicator=ind, background=0.,
                                 coordinate='degree', center_list=rf_center_list, sf_list=(0.04,),
                                 tf_list=(2.0,), dire_list=rf_direlist, con_list=(0.8,), radius_list=(20.,),
                                 block_dur=rf_block_dur, midgap_dur=rf_midgapdur, iteration=rf_iterationN, pregap_dur=5.,
                                 postgap_dur=2., is_blank_block=False, is_random_start_phase=False)
# ds.set_stim(rf)
# ds.trigger_display()


# ================ Receptive field mapping - flashing stimuli ================ 
fl_duration = 0.250
fl_midgap_dur = 2.
fl_color = (-0.8, 0.8)
fl_nreps = 15 

fl_pregap_dur = 5.
fl_postgap_dur = 2.

#test
#fl_duration = 1
#fl_midgap_dur = 1

fl = stim.RandomizedUniformFlashes(
    monitor=mon,
    indicator=ind,
    flash_dur=fl_duration,    
    midgap_dur=fl_midgap_dur,    
    n_reps=fl_nreps,            
    colors=fl_color,   
    pregap_dur=fl_pregap_dur,
    postgap_dur=fl_postgap_dur,
    background=0.0,
    coordinate='degree',
    rng_seed=1234,         # reproducible order
)
# ds.set_stim(fl)
# ds.trigger_display()


# ================ full-field Drifting Gratings ================ 
dg_direlist = (0.,45.,90.,135.,180.,225.,270.,315.,)
dg_tflist = (1.,2.,4.,8.,15.,)
dg_sflist = (0.04,)
dg_iterationN = 20
dg_block_dur = 2.
dg_midgapdur = 1.
# 30+ minutes...

# # test:
# dg_direlist = (0.,45.,)
# dg_tflist = (1.,)
# dg_sflist = (0.04,)
# dg_iterationN = 2
# dg_block_dur = 2.
# dg_midgapdur = 1.

dg = stim.DriftingGratingCircle(monitor=mon, indicator=ind, background=0.,
                                 coordinate='degree', center=(0., 60.), sf_list=dg_sflist,
                                 tf_list=dg_tflist, dire_list=dg_direlist, con_list=(0.8,), radius_list=(120.,),
                                 block_dur=dg_block_dur, midgap_dur=dg_midgapdur, iteration=dg_iterationN, pregap_dur=5.,
                                 postgap_dur=2., is_blank_block=True, is_random_start_phase=False)

# ds.set_stim(dg)
# ds.trigger_display()


# ================ full-field Static Gratings ================ 
sg_orilist = (0.,30.,60.,90.,120.,150.,)
sg_sflist = (0.02,0.04,0.08,0.16,0.32,)
sg_phaselist = (0.,0.25,0.50,0.75)
sg_iterationN = 50
sg_display_dur = 0.250
sg_midgapdur = 0.
sg_is_blank_block=True
# 25+ minutes...

# # test:
# sg_orilist = (0.,150.,)
# sg_sflist = (0.02,0.32,)
# sg_phaselist = (0.5,)
# sg_iterationN = 1
# sg_display_dur = 5
# sg_is_blank_block=True

sg = stim.StaticGratingCircle(monitor=mon, indicator=ind, background=0.,
                                coordinate='degree', center=(0., 60.), sf_list=sg_sflist,
                                ori_list=sg_orilist, con_list=(0.8,), radius_list=(120.,), 
                                display_dur=sg_display_dur, midgap_dur=sg_midgapdur, iteration=sg_iterationN, pregap_dur=5.,
                                postgap_dur=2., is_blank_block=sg_is_blank_block, phase_list = sg_phaselist,)

# ds.set_stim(sg)
# ds.trigger_display()


# ============== Static Images ===================================
si_img_center = (0., 60.)
si_deg_per_pixel = (0.5, 0.5)
si_display_dur = 0.250
si_midgap_dur = 0.
si_iteration = 50
si_is_blank_block = True
si_images_folder = os.path.join(os.path.dirname(rm.__file__), 'staticImages')
# 25 minutes...


si = stim.StaticImages(monitor=mon, indicator=ind, pregap_dur=5.,
                       postgap_dur=2., coordinate='degree',
                       background=0., img_center=si_img_center,
                       deg_per_pixel=si_deg_per_pixel, display_dur=si_display_dur,
                       midgap_dur=si_midgap_dur, iteration=si_iteration,
                       is_blank_block=si_is_blank_block)

print ('wrapping images ...')
static_images_path = os.path.join(si_images_folder, 'wrapped_images_for_display.hdf5')
if os.path.isfile(static_images_path):
    os.remove(static_images_path)
si.wrap_images(si_images_folder)


# # ======================= Stimulus Separator ======================================
# ss_indicator_on_frame_num = 4
# ss_indicator_off_frame_num = 4
# ss_cycle_num = 10
# ss = stim.StimulusSeparator(monitor=mon, indicator=ind, pregap_dur=2.,
#                             postgap_dur=2., coordinate='degree',
#                             background=0.,
#                             indicator_on_frame_num=ss_indicator_on_frame_num,
#                             indicator_off_frame_num=ss_indicator_off_frame_num,
#                             cycle_num=ss_cycle_num)
# =================================================================================

# ======================= Combined Stimuli ========================================
cs_stim_ind_sequence = [0, 1, 2, 3, 4]
cs = stim.CombinedStimuli(monitor=mon, indicator=ind, pregap_dur=2.,
                          postgap_dur=2., coordinate='degree',
                          background=0.)
# =================================================================================

# ======================= Set Stimuli Sequence ====================================
all_stim = [rf, fl, dg, sg, si]
stim_seq = [all_stim[stim_ind] for stim_ind in cs_stim_ind_sequence]
cs.set_stimuli(stimuli=stim_seq, static_images_path=static_images_path)
# =================================================================================

# =============================== display =========================================
ds.set_stim(cs)
log_path, log_dict = ds.trigger_display()
print(log_path)

# =============================== display =========================================


# ================ other ================ 

# # other - change

# uc = stim.UniformContrast(mon, ind, duration=10., color=-1.)
# ss = stim.StimulusSeparator(mon, ind)
# cs = stim.CombinedStimuli(mon, ind)
# cs.set_stimuli([ss, uc, ss])
# ds = DisplaySequence(log_dir='C:/data')
# # ds = DisplaySequence(log_dir='/home/zhuangjun1981')
# ds.set_stim(cs)
# log_path, log_dict = ds.trigger_display()



# ================ other ================ 

# # other - change

# uc = stim.UniformContrast(mon, ind, duration=10., color=-1.)
# ss = stim.StimulusSeparator(mon, ind)
# cs = stim.CombinedStimuli(mon, ind)
# cs.set_stimuli([ss, uc, ss])
# ds = DisplaySequence(log_dir='C:/data')
# # ds = DisplaySequence(log_dir='/home/zhuangjun1981')
# ds.set_stim(cs)
# log_path, log_dict = ds.trigger_display()
