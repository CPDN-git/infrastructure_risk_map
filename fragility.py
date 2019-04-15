###### plotting fragility curves for lines and towers

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


##### the fragility paper uses hourly wind speeds :(




# towers 
# normal cdf

def tower_prob(v,beta=0.25,S_ds=80,bottom=40,top=160):
	
#	prob=np.empty(shape=v.shape)
#	idx_low=np.where(v<=bottom)
#	idx_mid=np.where((v > bottom) & (v <= top))
#	idx_high=np.where(v>top)
#	
#	
#	prob[idx_low]=0
#	prob[idx_mid]=norm.cdf(np.log(v[idx_mid]/(S_ds))/beta)
#	prob[idx_high]=1
	
	prob=norm.cdf(np.log(v/(S_ds))/beta)
	
	#mean=np.sum(prob)/len(prob.flatten())
	
	return prob
	
def line_prob(v,bottom=30,top=60):
	
	prob=np.empty(shape=v.shape)
	idx_low=np.where(v<=bottom)
	idx_mid=np.where((v > bottom) & (v <= top))
	idx_high=np.where(v>top)
	
	
	prob[idx_low]=10**-2
	prob[idx_mid]=-1 + ((v[idx_mid]/(top-bottom)))
	prob[idx_high]=1
	
	#mean=np.sum(prob)/len(prob.flatten())
			
	return prob
	
def line_fragility_for_grid(data_wind):
	data_wind=np.squeeze(data_wind)
	line_probs=np.empty(shape=(data_wind.shape[2],data_wind.shape[3]))
	
	for lat_index in range(data_wind.shape[2]):
		for long_index in range(data_wind.shape[3]):
			line_probs[lat_index,long_index]=line_prob(data_wind[:,:,lat_index,long_index])
			print line_probs.shape
			print line_probs[lat_index,long_index]
			print(str(lat_index) + ',' + str(long_index))
	return line_probs		
				
def tower_fragility_for_grid(data_wind):
	data_wind=np.squeeze(data_wind)
	tower_probs=np.empty(shape=(data_wind.shape[2],data_wind.shape[3]))
	
	for lat_index in range(data_wind.shape[2]):
		for long_index in range(data_wind.shape[3]):
			tower_probs[lat_index,long_index]=tower_prob(data_wind[:,:,lat_index,long_index])
			print tower_probs.shape
			print tower_probs[lat_index,long_index]
			print(str(lat_index) + ',' + str(long_index))
	return tower_probs	

def main():
	
	#data_wind_hist=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_646_data_item3249_daily_mean_full_field.npy').squeeze()
	#data_wind_15=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_647_data_item3249_daily_mean_full_field.npy').squeeze()
	data_wind_2=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_648_data_item3249_daily_mean_full_field.npy').squeeze()				
		
	print np.amax(data_wind_2)
	line_frag_2=line_fragility_for_grid(data_wind_2)
	
	print line_frag_2.shape
	print np.amax(line_frag_2)
	
	tower_frag_2=tower_fragility_for_grid(data_wind_2)
	
		
				
	
	
def testing():
	x_values=np.arange(0,180,0.2)
	beta=0.23
	S_ds=80

	tower_probs=tower_prob(v=x_values,beta=beta,S_ds=S_ds)
	

	bottom=30
	top=60
	line_probs=line_prob(v=x_values,bottom=bottom,top=top)
	print len(line_probs)
	print len(tower_probs)

	line_probs=np.array(line_probs)
	tower_probs=np.array(tower_probs)

	np.save('line_fragility.npy',line_probs)
	np.save('tower_fragility.npy',tower_probs)

	f,ax = plt.subplots(1)
	f.suptitle('Fragility curve for L2 transmission tower')
	plt.plot(x_values,tower_probs,axes=ax)
	#plt.plot(x_values,line_probs,axes=ax)
	plt.xlabel('Wind Speed m/s')
	plt.ylabel('Probability of Failure')
	plt.show()

if __name__ == '__main__':
	#main()
	testing()



		
	
