## Turning daily mean wind speed into 24 lots of hourly speeds



import numpy as np
import math as m


## Curently only been used on field meaned data
data_648_wind=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_648_data_item3249_daily_mean.npy').flatten()

data_647_wind=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_647_data_item3249_daily_mean.npy').flatten()

data_646_wind=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_646_data_item3249_daily_mean.npy').flatten()

batches=[data_646_wind,data_647_wind,data_648_wind]

#data_648_wind=data_648_wind.flatten()
#print data_648_wind.shape

#print data_648_wind

for batch in batches:
	
	hourly_wind=np.empty(batch.shape[0])
	print batch.shape

	# iteratre through hours of day, starting at 00:00, ending at 23:00
	for t in range(0,24):
		#initialise empty colmun of length equal to raw data
		hours=np.empty(batch.shape[0])

		#fill column with values corresponsing to sinusoidal oscillation about
		#the mean value from the batch
		hours=batch*(1 + 0.5*m.cos(t*m.pi/12))
		
		#append the new set of hourly values, to the growing set of previously
		#calculated ones
		hourly_wind=np.column_stack([hourly_wind,hours])
		

	#remove initial empty row
	hourly_wind=np.delete(hourly_wind,0,1)

	
	if np.array_equal(batch,data_648_wind):
		np.save('batch_648_data_item3249'  + '_hourly.npy',np.array(hourly_wind))
		print('648 saved')
	elif np.array_equal(batch,data_647_wind):
		np.save('batch_647_data_item3249'  + '_hourly.npy',np.array(hourly_wind))
		print('647 saved')
	elif np.array_equal(batch,data_646_wind):
		np.save('batch_646_data_item3249'  + '_hourly.npy',np.array(hourly_wind))
		print('646 saved') 

	print hourly_wind.shape


