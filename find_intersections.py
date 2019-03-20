##### takes in set of lines making up grid and finds points of intersections with
##### other line shapefile by using points of minimum distance

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import GeometryCollection
from shapely.ops import split
import numpy as np
import shapely
import matplotlib.pyplot as plt
import random
from plot_grid_v3 import gen_grid_v3
from plot_grid_v3 import gen_line_grid


def overlapped_lines(linefile_gdf,box_gdf):
	
	##all strings connected to box
	temp=gpd.sjoin(linefile_gdf,box_gdf,how='inner',op='intersects')
	
	
	##empty gdf
	overlap=gpd.GeoDataFrame()
	
	#### iterate over geometries of linestrings colliding with the box, line at
	#### a time
	for geom in temp['geometry']:
		
		# make a single row dataframe, with the overlap between the current
		# line and the box
		line_overlap=gpd.GeoDataFrame(box_gdf.intersection(geom))

		# append above row onto the empty dataframe
		overlap=overlap.append(line_overlap)
	
	#
	#print overlap	
	# create geometry column for overlap frame, made of its old values (geometries 
	# of box-line overlap)
	if overlap.empty:
		return overlap
	else:
		overlap['geometry']=overlap
	
	return overlap
	
def output_weights(lines_gdf):
	if lines_gdf.empty:
		return 0
	
	if lines_gdf['geometry'].iloc[0].geom_type =='Point':
		num=lines_gdf.shape[0]
		return num
		
	if lines_gdf['geometry'].iloc[0].geom_type =='Polygon':
		area=lines_gdf['geometry'].area.sum()
		return area
		
	
	length=lines_gdf['geometry'].length.sum()
	return length

	
	

def main():

	shape_dir = '/home/users/jburns59/NatGridShapes/'
	OHL_link = shape_dir + 'OHLs/'
	OHLs_400 = gpd.read_file(OHL_link + '400kV/OHLs_400.shp')


	line=[LineString([Point(-5,50),Point(-3,52)])]
	line_gdf=gpd.GeoDataFrame({'geometry':line})
	line_gdf.crs=OHLs_400.crs


	### building polygon box

	x=-4
	y=51
	height=3
	width=2

	box=[Polygon([(x-(width/2),y+(height/2)), (x+(width/2),y+(height/2)), (x+(width/2),y-(height/2)), (x-(width/2), y-(height/2))])]
	box_gdf=gpd.GeoDataFrame({'geometry':box})
	box_gdf.crs=OHLs_400.crs

	
	#### test plots the entirety of any linestrings that touch the box

	test= gpd.sjoin( OHLs_400,box_gdf, how="inner", op='intersects')



	#### GRID read
	#grid = gpd.read_file('grid.shp')
	
	#### empty gdf, currently no col names
	overlap=gpd.GeoDataFrame()	
	

	#### iterate over geometries of linestrings colliding with the box, line at
	#### a time
	for geom in test['geometry']:
		
		# make a single row dataframe, with the overlap between the current
		# line and the box
		line_overlap=gpd.GeoDataFrame(box_gdf.intersection(geom))

		# append above row onto the empty dataframe
		overlap=overlap.append(line_overlap)
		
	
		
	# create geometry column for overlap frame, made of its old values (geometries 
	# of box-line overlap)
	overlap['geometry']=overlap

	print overlap.head(5)
	
	overlap=overlapped_lines(OHLs_400,box_gdf)

	len1=test['geometry'].length.sum()
	len2=overlap['geometry'].length.sum()
	
		

	f,ax = plt.subplots(1)
	line_gdf.plot(axes=ax,color='blue',alpha=0.7,markersize=3)
	test.plot(axes=ax,color='yellow',alpha=0.7)
	box_gdf.plot(axes=ax,alpha=0.5)
	overlap.plot(axes=ax,color='red',alpha=0.7)

	plt.show()

if __name__ == "__main__":
	main()


