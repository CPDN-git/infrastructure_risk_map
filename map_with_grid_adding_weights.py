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
from find_intersections import output_weights,overlapped_lines


def add_column(df,name_string,values):
	df[name_string]=values
	return df


PYTHONHASHSEED=0

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
#df_edges = gpd.read_file(df_edges_link)



#######################################

#reference points

cities = pd.DataFrame({'City':['London','Birmingham','Manchester','Leeds','Newcastle'],'Country':['UK','UK','UK','UK','UK'],'Latitude':[51.51279,52.48667,53.478904,53.802122,54.975496],'Longitude':[-0.09184,-1.89194,-2.24395,-1.549483,-1.614543]})

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
#df_edges = df_edges.to_crs(cities_gdf.crs)
OHLs=OHLs.to_crs(cities_gdf.crs)

#print subs

lenny=OHLs['geometry'].length.sum()





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


###### making polygon grid
#grid=gen_grid_v3(xmin=-5,ymin=50.125,xmax=1.75,ymax=54.875,width=0.25,height=0.25)

grid=gpd.read_file('grid_w_weights.shp')
grid.crs=cities_gdf.crs

print grid.iloc[0]

############### finding amount of infrastructure within each grid

frames=[subs]

for frame in frames:
	weights=[]
	print (frame.get_label() + '\n')
	for r in range(grid.shape[0]):
		weights+=[output_weights(overlapped_lines(box_gdf=grid.iloc[[r]],linefile_gdf=frame.get_frame()))]
	grid=add_column(grid,frame.get_label(),weights)	
	print grid.to_string()	

#grid.to_file('grid_w_weights.shp',driver='ESRI Shapefile')

print grid[subs.get_label()]

 





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


frames=[towers,OHLs_275,OHLs_132,OHLs_400,britain,subs]


patches=[]

f, (ax, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)


for dataframe in frames:
	## looping over plottable frame objects and plotting
	dataframe.get_frame().plot(axes=ax,alpha=dataframe.get_alpha(),color=dataframe.get_colour(),markersize=dataframe.get_size())
	## checking for label, and if it has one, creating a legend patch for it
	if dataframe.get_label() != '':
		patches+=[mpatches.Patch(color=dataframe.get_colour(),label=dataframe.get_label())]


for x, y, label in zip(cities_gdf.geometry.x, cities_gdf.geometry.y, cities_gdf.City):
	ax.annotate(label, xy=(x,y), xytext=(3,3), textcoords="offset points")
cities_gdf.plot(axes=ax,color='black',markersize=1)


grid.plot(axes=ax,column=subs.get_label(),cmap='Greys',alpha=0.7)
grid.plot(axes=ax2,column=subs.get_label(),cmap='Greys',alpha=1)


f.suptitle('Nat Grid Overhead Lines, Towers, and Substations')
plt.legend(handles=patches)
plt.savefig('map_with_split_OHLs.png')
plt.show()





