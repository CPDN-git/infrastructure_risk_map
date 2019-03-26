###### Making risk ratios for each grid cell, storing them
###### in numpy array ready to be added to grid

import numpy as np
import pandas as pd


from find_threshold_prob import prob_below 
from find_threshold_prob import prob_above 

def cell_risk_ratio(data,dataHist,lat_index,long_index,threshold,below):
	if below == 'below':
		P=prob_below(data[:,:,:,lat_index,long_index],threshold)
		print P
		PHist=prob_below(dataHist[:,:,:,lat_index,long_index],threshold)
		print PHist
		if PHist==0:
			if P == PHist:
				ratio=1
			else:
				PHist=-1.8*10**-11
				ratio=float(P/PHist)
		else:
			ratio=float(P/PHist)
		return ratio
	else:
		P=prob_above(data[:,:,:,lat_index,long_index],threshold)
		print P
		PHist=prob_above(dataHist[:,:,:,lat_index,long_index],threshold)
		print PHist
		if PHist==0:
			if P == PHist:
				ratio=1
			else:
				PHist=-1.8*10**-11
				ratio=P/PHist
		else:
			ratio=float(P/PHist)
		return ratio
	
def grid_ratios(batches,dataHist,threshold,below):
	
	## initialising empty numpy array, 1D of length == no. of grid cells
	ratio_array=np.empty(dataHist.shape[3]*dataHist.shape[4])
	print ratio_array.shape
	
	
	for batch in batches:
		print('new batch')
		ratios=[]	
		
		## iterating over grid cells, going along longitude axis
		for lat_index in range(batch.shape[3]):
			for long_index in range(batch.shape[4]):
				
				## for current batch and grid cell, calculate risk ratio for threshold across all ensembles
				ratio=cell_risk_ratio(batch,dataHist,lat_index,long_index,threshold,below)
				ratios+=[ratio]
				print ('ratio for ' + str(lat_index) +',' + str(long_index)+
				 ' = ' + str(ratio))
		
		## make array of all ratios from list		 
		ratios=np.array(ratios)
		
		## append array column to empty column
		ratio_array=np.column_stack([ratio_array,ratios])
	
	## remove the empty arrray column
	ratio_array=np.delete(ratio_array,0,1)	
	return ratio_array
	
	
def main():
	dataTmaxHist=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_646_data_item3236_daily_maximum_full_field.npy')
	dataTmax15=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_647_data_item3236_daily_maximum_full_field.npy')
	dataTmax2=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_648_data_item3236_daily_maximum_full_field.npy')
	batches=[dataTmax15,dataTmax2]
	
	array_of_ratios=grid_ratios(batches,dataTmaxHist,298,'')
	np.save('risk_ratio_above_298K',array_of_ratios)
	
if __name__=='__main__':
	main()
	
