## For each temperature value, there is a corresponding threshold of 
## wind speed to cause power reduction. For each of these threshold 
## wind speeds, there is a certain probability that the wind is below
## that speed (dependent on which world we look at). 

import numpy as np


def prob_below(dataHist,threshold):
	
	dataHist=dataHist.flatten()
	HistCount=sum(np.array(dataHist)<threshold)
	print HistCount
	#NatCount=sum(np.array(dataNat)<threshold)
	dataHist=dataHist.flatten()
	#i=0
	#for datum in dataHist:
	#	if datum < threshold:
	#		i+=1
	#HistCount=i
	
	P_hist=float(HistCount)/float(len(dataHist))
	#P_nat=float(NatCount)/float(len(dataNat))
	#RiskRatio=P_hist/P_nat
	
	return P_hist
	


def main():
	#getting thresholds of damage for cables
	hv_new_threshold=np.load('hv_new_threshold.npy')
	hv_old_threshold=np.load('hv_old_threshold.npy')
	lv_old_threshold=np.load('lv_old_threshold.npy')

	#1st col is temp, 2nd is wind speed
	

	
	#print(hv_new_threshold)
	#print(hv_new_threshold.shape[0])


	#now need data
	

	### HOURLY DATA
        data=np.load('/home/users/jburns59/tmax_vs_wind_distributions/batch_647_data_item3249_hourly.npy')
	#data_648_wind=data_648_wind.flatten()

	#data_646_wind=np.load('/home/users/jburns59/tmax_vs_wind_distributions/batch_646_data_item3249_hourly.npy')

	#print np.amin(data_648_wind)

	#### DAILY MEAN DATA
	#data_648_wind=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_648_data_item3249_daily_mean.npy')
	
	
        
	
	cables=[hv_old_threshold,lv_old_threshold,hv_new_threshold]
	for cable_type in cables:
		prob_values=[]
		for T in range(cable_type.shape[0]):
			prob_values+=[prob_below(data,cable_type[T,1])]
	
		cable_type_new=np.column_stack([cable_type,prob_values])
		
		if np.array_equal(cable_type,hv_new_threshold):
			hv_new_threshold=cable_type_new
			np.save('hv_new_threshold_wind_probs_15deg_summer.npy',hv_new_threshold)
			print('saved new')
		elif np.array_equal(cable_type,hv_old_threshold):
			hv_old_threshold=cable_type_new
			np.save('hv_old_threshold_wind_probs_15deg_summer.npy',hv_old_threshold)
			print('saved old hv')
		elif np.array_equal(cable_type,lv_old_threshold):
			lv_old_threshold=cable_type_new
			np.save('lv_old_threshold_wind_probs_15deg_summer.npy',lv_old_threshold)
			print('saved old lv')
		print cable_type
	
		

main()

