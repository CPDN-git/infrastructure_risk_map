## plotting some natGrid shapefiles


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import numpy as np
from geopandas.tools import sjoin
import pysal as ps
from pysal.contrib.viz import mapping as maps
from shapely.geometry import Point
from plot_grid_v3 import gen_grid_v3
from plot_grid_v3 import plottable_frame
from linestring_grid import line_grid
import shapely.ops
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely import wkt
from plot_grid_v3 import gen_line_grid
from find_intersections import output_weights,overlapped_lines,average_height
from wind_vs_height import wind_scale_for_grid

def add_column(df,name_string,values):
	df[name_string]=values
	return df


##################################
#importing natGrid shapefiles

shape_dir = '/home/users/jburns59/NatGridShapes/'

OHL_link = shape_dir + 'OHLs/'
OHLs_400 = gpd.read_file(OHL_link + '400kV/OHLs_400.shp')
OHLs_275 = gpd.read_file(OHL_link + '275kV/OHLs_275.shp')
OHLs_132 = gpd.read_file(OHL_link + '132kV/OHLs_132.shp')
OHLs = gpd.read_file(shape_dir + 'OHL.shp')

tower_link = shape_dir + 'Towers/Towers.shp'
towers = gpd.read_file(tower_link)

#cable_link = shape_dir + 'Cables/Cable.shp'
#cables = gpd.read_file(cable_link)

df_nodes_link = shape_dir + 'Substations/National/Substations.shp'
subs = gpd.read_file(df_nodes_link)

df_dno_link = shape_dir + 'Britain/df_dno.shp'
britain = gpd.read_file(df_dno_link)

#df_edges_link = shape_dir + 'OHLs/11kv_NW/df_edges_gr11kv.shp'
#OHLs_11 = gpd.read_file(df_edges_link)



#######################################

#reference points

cities = pd.DataFrame({'City':['London','Birmingham','Manchester','Leeds','Newcastle','Cardiff','Plymouth'],'Country':['UK','UK','UK','UK','UK','UK','UK'],'Latitude':[51.51279,52.48667,53.478904,53.802122,54.975496,51.48,50.376289],'Longitude':[-0.09184,-1.89194,-2.24395,-1.549483,-1.614543,-3.18,-4.143841]})

cities['Coordinates'] = list(zip(cities.Longitude,cities.Latitude))
cities['Coordinates'] = cities['Coordinates'].apply(Point)

cities_gdf = gpd.GeoDataFrame(cities,geometry='Coordinates')
cities_gdf.crs = {'init' : 'epsg:4326'}


## Aligning CRSs to cities dataframe

OHLs_400.crs = cities_gdf.crs
OHLs_275.crs = cities_gdf.crs
OHLs_132.crs = cities_gdf.crs
towers = towers.to_crs(cities_gdf.crs)
#cables = cables.to_crs(cities_gdf.crs)
britain = britain.to_crs(cities_gdf.crs)
subs = subs.to_crs(cities_gdf.crs)
#OHLs_11 = OHLs_11.to_crs(cities_gdf.crs)
OHLs=OHLs.to_crs(cities_gdf.crs)

#print subs

#lenny=OHLs['geometry'].length.sum()





#  Saving new GeoDataFrames as shp files

#OHLs_400.to_file(driver = 'ESRI Shapefile', filename = '/home/users/jburns59/NatGridShapes/OHLs/400kV/OHLs_400.shp')
#OHLs_275.to_file(driver = 'ESRI Shapefile', filename = '/home/users/jburns59/NatGridShapes/OHLs/275kV/OHLs_275.shp')
#OHLs_132.to_file(driver = 'ESRI Shapefile', filename = '/home/users/jburns59/NatGridShapes/OHLs/132kV/OHLs_132.shp')


#######################################
# Instantiating plottable frames


OHLs_400=plottable_frame(frame=OHLs_400,label='OHLs 400kV'
,colour='yellow',alpha=0.7,size=4)

OHLs_275=plottable_frame(frame=OHLs_275,label='OHLs 275kV'
,colour='red',alpha=0.7,size=4)

OHLs_132=plottable_frame(frame=OHLs_132,label='OHLs 132kV'
,colour='orange',alpha=0.7,size=4)

towers=plottable_frame(frame=towers,label='Towers'
,colour='blue',alpha=0.7,size=2)

britain=plottable_frame(frame=britain,label=''
,colour='gold',alpha=0.3)

subs=plottable_frame(frame=subs,label='Substations'
,colour='green',alpha=1)

#OHLs_11=plottable_frame(frame=OHLs_11,label='OHLs 11kV'
#,colour='green',alpha=0.7)


###### making polygon grid
#grid=gen_grid_v3(xmin=-5,ymin=50.125,xmax=1.75,ymax=54.875,width=0.25,height=0.25)

grid=gpd.read_file('grid_w_weights.shp')
grid.crs=cities_gdf.crs

print grid.iloc[0]

############### finding amount of infrastructure within each grid

frames=[]


for frame in frames:
	print('\ngetting weights')
	weights=[]
	print(grid.shape[0])
	print (frame.get_label() + '\n')
	for r in range(grid.shape[0]):
		print(r)
		weights+=[output_weights(overlapped_lines(box_gdf=grid.iloc[[r]],linefile_gdf=frame.get_frame()))]
	grid=add_column(grid,frame.get_label(),weights)	
	
print('Done with weights!')	


#grid.to_file('grid_w_weights.shp',driver='ESRI Shapefile')


################ finding representative heights



frames=[]

for frame in frames:
	print('\ngetting heights')
	heights=[]
	print(grid.shape[0])
	print (frame.get_label() + '\n')
	for r in range(grid.shape[0]):
		print(r)
		heights+=[average_height(box_gdf=grid.iloc[[r]],towers_gdf=frame.get_frame())]
	#grid=add_column(grid,'height',heights)	
	
print('Done with heights!')

#grid['height']=heights





########### Addding wind scaling for differentivate heights
print grid.wind_scale


scales=wind_scale_for_grid(grid,measurement_height=10)
print np.amax(scales)
print np.amin(scales[scales>1])
#grid['wind_scale']=scales

for r in range(grid.shape[0]):
	print(str(grid.height.iloc[r]) + '		' + str(grid['wind_scale'].iloc[r]))

grid.to_file('grid_w_weights.shp',driver='ESRI Shapefile')

########## Adding risk ratios to grid TRIAL

dataRisk=np.load('risk_ratio_above_298K_new.npy')

grid['risk_1.5']=dataRisk[:,0]
grid['risk_2']=dataRisk[:,1]

print grid
print grid['risk_1.5']
print grid['risk_2']

print np.amax(dataRisk,0)

x=np.array(dataRisk[:,0])
vals=x[x>1]
min_15=np.amin(vals)

x=np.array(dataRisk[:,1])
vals=x[x>1]
min_2=np.amin(vals)

print min_15
 





#print weights
#bins=5
#top=np.amax(weights)
#print top
#bottom=0
#bounds = np.linspace(bottom,top,bins+1)
#print bounds
#bins_values=np.array([float(0)]*(bins+1))

#norm_weights=[]

#for weight in weights:
#	for index in range(len(bounds)-1):
#		if weight >= bounds[index] and weight < bounds[index+1]:
#			norm_weights+=[index]

#norm_weights+=[0]
#print len(weights)
#print len(norm_weights)



#grid['vals']=weights

###### List of frames you want to plot, excluding the grid
frames=[towers,OHLs_275,OHLs_132,OHLs_400,britain,subs]


patches=[]

print('\nPlotting')

f, (ax, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
axes=[ax,ax2]

## counter to prevent stack of patches in legend
i=0
for a in axes:
	for dataframe in frames:
		## looping over plottable frame objects and plotting
		dataframe.get_frame().plot(axes=a,alpha=dataframe.get_alpha(),color=dataframe.get_colour(),markersize=dataframe.get_size())
		## checking for label, and if it has one, creating a legend patch for it
		if dataframe.get_label() != '' and i==0:
			patches+=[mpatches.Patch(color=dataframe.get_colour(),label=dataframe.get_label())]
	i=1

	for x, y, label in zip(cities_gdf.geometry.x, cities_gdf.geometry.y, cities_gdf.City):
		a.annotate(label, xy=(x,y), xytext=(3,3), textcoords="offset points")
	cities_gdf.plot(axes=ax,color='black',markersize=1)


cmap = plt.cm.plasma
cmap.set_under(color='white')


grid.plot(axes=ax,column='risk_1.5',cmap=cmap,alpha=0.7,legend=True,vmin=min_15)
grid.plot(axes=ax2,column='risk_2',cmap=cmap,alpha=0.7,legend=True,vmin=min_2)


f.suptitle('Map of risk ratio for temperatures above 25C')

ax.set(xlabel='Longitude',ylabel='Latitude')
ax2.set(xlabel='Longitude',ylabel='Latitude')
ax.set_title('1.5 degree warming')
ax2.set_title('2 degree warming')



plt.legend(handles=patches)
plt.savefig('25deg_risk.png')
plt.show()





