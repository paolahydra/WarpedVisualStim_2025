import os
import numpy as np
import matplotlib.pyplot as plt
import WarpedVisualStim as rm
import WarpedVisualStim.StimulusRoutines as stim
from WarpedVisualStim.DisplayStimulus import DisplaySequence
from WarpedVisualStim.MonitorSetup import Monitor, Indicator


# Initialize Monitor object
mon_resolution = (1440, 2560)  # enter your monitors resolution
mon_width_cm = 60.  # enter your monitors width in cm
mon_height_cm = 34.  # enter your monitors height in cm
mon_refresh_rate = 60  # enter your monitors height in Hz
mon_C2T_cm = mon_height_cm / 2.		
mon_C2A_cm = mon_width_cm / 2.		
mon_center_coordinates = (0., 60.) 	#these should be the angular coordinates of the center of the screen with respect to mouse
mon_dis = 15.
mon_downsample_rate = 5 		#what is this?

mon = Monitor(resolution=mon_resolution, dis=mon_dis, mon_width_cm=mon_width_cm,
              mon_height_cm=mon_height_cm, C2T_cm=mon_C2T_cm, C2A_cm=mon_C2A_cm,
              center_coordinates=mon_center_coordinates,
              downsample_rate=mon_downsample_rate)


# Initialize indicator
ind_width_cm = 1.
ind_height_cm = 1.
ind_position = 'southeast'
ind_is_sync = True
ind_freq = 1 

ind = Indicator(mon, width_cm=ind_width_cm, height_cm=ind_height_cm,
                position=ind_position, is_sync=ind_is_sync, freq=ind_freq)


# General data settings
ds_log_dir = r'D:\data\displaySequence'
ds_backupdir = None
ds_identifier = 'TEST'
ds_display_iter = 1
ds_mouse_id = 'MOUSE'
ds_user_id = 'USER'
ds_psychopy_mon = 'testMonitor'
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
ds_display_screen = 1
ds_initial_background_color = 0.
ds_color_weights = (1., 1., 1.)

# ================ Initialize the DisplaySequence object ==========================
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


# ================ Receptive field mapping - drifting gratings gabor ================ 

# input center parameters
x_range = (20.0, 100.0)   # [xmin, xmax]
y_range = (15.0, -15.0)   # preserves given orientation (top→bottom)
n = 9                     # how much you want to tile each axis

# calculate grid of centers
x_edges = np.linspace(x_range[0], x_range[1], n + 1)
y_edges = np.linspace(y_range[0], y_range[1], n + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
# n x n grid of centers, then flatten to list of tuples (x,y)
Xc, Yc = np.meshgrid(x_centers, y_centers, indexing='xy')  # shape (9,9)

rf_center_list = list(map(tuple, np.column_stack([Xc.ravel(), Yc.ravel()])))

rf_direlist = (0.,90.,180.,270., 45.,135.,225.,315.,)
rf_iterationN = 8
rf_direlist = (0.,90.,180.,270.,)
rf_iterationN = 15
# both options last roughly 20-21 minutes

#test:
rf_center_list = [(10., 20.),(-10., 100.), (-10., 20.),(10., 100.)]
rf_direlist = (0.,90.,180.,270.,)
rf_iterationN = 1

rf = stim.DriftingGratingMultipleCircle(monitor=mon, indicator=ind, background=0.,
                                 coordinate='degree', center_list=rf_center_list, sf_list=(0.04,),
                                 tf_list=(2.0,), dire_list=rf_direlist, con_list=(0.8,), radius_list=(20.,),
                                 block_dur=0.250, midgap_dur=0., iteration=rf_iterationN, pregap_dur=2.,
                                 postgap_dur=2., is_blank_block=False, is_random_start_phase=False)
# ds.set_stim(rf)
# ds.trigger_display()


# ================ Receptive field mapping - flashing stimuli ================ 
fl_duration = 0.250
fl_color = (-0.8, 0.8)
fl_pregap_dur = 2
fl_postgap_dur = 2
fl_midgap_dur = 0.250
fl_reps = 5

fl = stim.RandomizedUniformFlashes(
    monitor=mon,
    indicator=ind,
    flash_dur=fl_duration,    
    midgap_dur=fl_midgap_dur,    
    n_reps=fl_reps,            
    colors=fl_color,   
    pregap_dur=fl_pregap_dur,
    postgap_dur=fl_postgap_dur,
    background=0.0,
    coordinate='degree',
    rng_seed=1234,         # reproducible order
    balance_colors=True    # enforce ~equal counts, then shuffle
)
# ds.set_stim(fl)
# ds.trigger_display()


# ================ full-field Drifting Gratings ================ 
dg_direlist = (0.,45.,90.,135.,180.,225.,270.,315.,)
dg_iterationN = 15
# test:
dg_direlist = (0.,45.,90.,135.,)
dg_iterationN = 5

dg = stim.DriftingGratingCircle(monitor=mon, indicator=ind, background=0.,
                                 coordinate='degree', center=(0., 60.), sf_list=(0.04,),
                                 tf_list=(1.,2.,4.,8.,15.,), dire_list=rf_direlist, con_list=(0.8,), radius_list=(120.,),
                                 block_dur=2., midgap_dur=1., iteration=rf_iterationN, pregap_dur=2.,
                                 postgap_dur=2., is_blank_block=True, is_random_start_phase=False)

# ds.set_stim(dg)
# ds.trigger_display()


# ================ full-field Static Gratings ================ 
sg_orilist = (0.,30.,60.,90.,120.,150.,)
sg_sflist = (0.02,0.04,0.08,0.16,0.32,)
sg_phaselist = (0.,0.25,0.50,0.75)
sg_iterationN = 50
# test:
sg_orilist = (0.,150.,)
sg_sflist = (0.02,0.32,)
sg_phaselist = (0.,0.50,)
sg_iterationN = 5

sg = stim.StaticGratingCircle(monitor=mon, indicator=ind, background=0.,
                                coordinate='degree', center=(0., 60.), sf_list=sg_sflist,
                                ori_list=sg_orilist, con_list=(0.8,), radius_list=(120.,), 
                                display_dur=0.250, midgap_dur=0., iteration=sg_iterationN, pregap_dur=2.,
                                postgap_dur=2., is_blank_block=True, phase_list = sg_phaselist,)

# ds.set_stim(sg)
# ds.trigger_display()


# ============== Static Images ===================================
si_img_center = (0., 60.)
si_deg_per_pixel = (0.5, 0.5)
si_display_dur = 0.250
si_midgap_dur = 0.
si_iteration = 2
si_is_blank_block = True
si_images_folder = os.path.join(os.path.dirname(rm.__file__), 'test', 'test_data')

si = stim.StaticImages(monitor=mon, indicator=ind, pregap_dur=2,
                       postgap_dur=2, coordinate='degree',
                       background=0., img_center=si_img_center,
                       deg_per_pixel=si_deg_per_pixel, display_dur=si_display_dur,
                       midgap_dur=si_midgap_dur, iteration=si_iteration,
                       is_blank_block=si_is_blank_block)

print ('wrapping images ...')
static_images_path = os.path.join(si_images_folder, 'wrapped_images_for_display.hdf5')
if os.path.isfile(static_images_path):
    os.remove(static_images_path)
si.wrap_images(si_images_folder)


# ======================= Stimulus Separator ======================================
ss_indicator_on_frame_num = 4
ss_indicator_off_frame_num = 4
ss_cycle_num = 10
ss = stim.StimulusSeparator(monitor=mon, indicator=ind, pregap_dur=2.,
                            postgap_dur=2., coordinate='degree',
                            background=0.,
                            indicator_on_frame_num=ss_indicator_on_frame_num,
                            indicator_off_frame_num=ss_indicator_off_frame_num,
                            cycle_num=ss_cycle_num)
# =================================================================================

# ======================= Combined Stimuli ========================================
cs_stim_ind_sequence = [0, 1, 2, 3, 4, 5]
cs = stim.CombinedStimuli(monitor=mon, indicator=ind, pregap_dur=2.,
                          postgap_dur=2., coordinate='degree',
                          background=0.)
# =================================================================================

# ======================= Set Stimuli Sequence ====================================
all_stim = [rf, fl, dg, sg, si, ss]
stim_seq = [all_stim[stim_ind] for stim_ind in cs_stim_ind_sequence]
cs.set_stimuli(stimuli=stim_seq, static_images_path=static_images_path)
# =================================================================================

# =============================== display =========================================
ds.set_stim(cs)
log_path, log_dict = ds.trigger_display()
print(log_path)
print(log_dict)
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
