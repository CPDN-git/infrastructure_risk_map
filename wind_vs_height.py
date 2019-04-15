## wind_vs_height

import numpy as np
import matplotlib.pyplot as plt

def log_profile():
	
	# some heights
	top=60
	h_values=np.arange(0,top,0.2)
	z0=0.05
	zMeas=10
	
	hub_speed_over_meas_speed=np.log(h_values/z0)/np.log(zMeas/z0)
	hub_speed_over_meas_speed[hub_speed_over_meas_speed < 0]=0
	
	f,ax = plt.subplots(1)
	f.suptitle('Change in wind speed with respect to height')
	plt.plot(hub_speed_over_meas_speed,h_values,axes=ax)
	plt.xlabel('V/V_10m')
	plt.ylabel('Height m')
	plt.yticks(np.arange(0,top,10))
	plt.show()
	
def wind_scale_for_grid(grid_gdf,measurement_height):
	z0=0.3
	scale=[]
	for r in range(grid_gdf.shape[0]):
		scale+=[np.log(grid_gdf.height.iloc[r]/z0)/np.log(measurement_height/z0)]
	return scale
	
if __name__ == '__main__':
	log_profile()


	
