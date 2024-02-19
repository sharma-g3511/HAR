feat_temp_t="/home/lasii/Research/MM_EXT/NEW/t_a_1d.npy"
feat_temp_d="/home/lasii/Research/MM_EXT/NEW/d_a_1d.npy"
feat_spect_t="/home/lasii/Research/MM_EXT/NEW/t_a_2d.npy"
feat_spect_d="/home/lasii/Research/MM_EXT/NEW/d_a_2d.npy"


python attention_fusion.py --feat_spect_t $feat_spect_t --feat_temp_t $feat_temp_t\
 --feat_temp_d $feat_temp_d --feat_spect_d $feat_spect_d