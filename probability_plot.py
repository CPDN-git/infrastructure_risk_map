## plotting probability

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

hv_new=np.load('hv_new_threshold_wind_probs_2deg_summer.npy')
hv_old=np.load('hv_old_threshold_wind_probs_2deg_summer.npy')
lv_old=np.load('lv_old_threshold_wind_probs_2deg_summer.npy')



#plt.plot(cable_type[:,0],cable_type[:,2],color=col)
#plt.show()

cables=[hv_new,hv_old,lv_old]
patches = []

#f,ax = plt.subplots(1)

for cable_type in cables:
	if np.array_equal(cable_type,hv_new):
		col='blue'
		plt.plot(cable_type[:,0],cable_type[:,2],color=col)
		patch= mpatches.Patch(color=col,label='Gap Style New')
		patches+=[patch]
	elif np.array_equal(cable_type,hv_old):
		col='red'
		plt.plot(cable_type[:,0],cable_type[:,2],color=col)
		patch= mpatches.Patch(color=col,label='ACSR High Voltage Old')
		patches+=[patch]
	elif np.array_equal(cable_type,lv_old):
		col='green'
		plt.plot(cable_type[:,0],cable_type[:,2],color=col)
		patch= mpatches.Patch(color=col,label='ACSR Low Voltage Old')
		patches+=[patch]

#f.suptitle('Probability of power reduction in 2 degree summer')
plt.legend(handles=patches)
plt.show()
