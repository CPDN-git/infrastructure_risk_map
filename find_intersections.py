##### takes in set of lines making up grid and finds points of intersections with
##### other line shapefile by using points of minimum distance

import geopandas as gpd
import pandas as pd
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
	
def average_height(towers_gdf,box_gdf):
	
	## all towers within box
	temp=gpd.sjoin(towers_gdf,box_gdf,how='inner',op='intersects')
	
	
	if not temp.empty:
		counts=[]
		lines=[u'L6',u'L8',u'L2',u'L12',u'L7',u'WOOD',u'L3',u'L66']
		heights=[50.6,50.5,42,46.3,26.2,10,38,34]
		
		cum_height=0
		
		temp['LINE_SERIE']=temp['LINE_SERIE'].str.strip()
		
		
		maxi=0
		idx=''
		
		for line in lines:
			if line in temp.LINE_SERIE.values:
				
				count= (temp.LINE_SERIE == line).sum()
				counts+=[count]
				cum_height+=count*heights[lines.index(line)]
				
				if count > maxi:
					maxi=count
					idx=lines.index(line)
		
		
		towers_added=sum(counts)
		
		default=heights[idx]
		
		
		
		
		remaining=temp.shape[0]-towers_added
		
		
		cum_height+=remaining*default
					
		
		
		average=cum_height/temp.shape[0]
		
		
		return average
	else:
		return 0
				
		
		
		
	
	
	
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
	towers_link = shape_dir + 'Towers/'
	towers = gpd.read_file(towers_link + 'Towers.shp')

	#for r in range(towers.shape[0]): 
	#	print towers.iloc[r]
	
	
	print towers.LINE_SERIE
	#print towers.TOWER_CONS
	
	duplicates=[]
	for line in towers.LINE_SERIE:
		if line not in duplicates:
			duplicates += [line]
	
	
	line_series=[]
	tower_construction=[]
	number=[]
	
	for line in duplicates:
		print('\n')
		#print towers[towers.LINE_SERIE == line]
		tow_cons= towers.TOWER_CONS[towers.LINE_SERIE == line]
		
		#print towers.TOWER_CONS[towers.LINE_SERIE == line].shape[0]
		
		duplicate_towers=[]
		for r in range(tow_cons.shape[0]):
			if  tow_cons.iloc[r] not in duplicate_towers:
				duplicate_towers += [tow_cons.iloc[r]]
				line_series +=[line]
				tower_construction+=[tow_cons.iloc[r]]
		
	line_frame=pd.DataFrame({'Lineseries':line_series,'Tower_con':tower_construction})
	
	count=[]
	for r in range(line_frame.shape[0]):
		count+= [towers[(towers.LINE_SERIE == line_frame.Lineseries.iloc[r]) & (towers.TOWER_CONS == line_frame.Tower_con.iloc[r])].shape[0]]
	
	line_frame['number']=count	
	
	print line_frame	
	
	print line_frame.number.sum()
	print towers.shape[0]
			
	
	
	#line_frame.to_pickle('tower_heights.pkl')
	
	line_frame=line_frame.sort_values(by=['Lineseries','number'],ascending=False)
	
	print line_frame
	
	print duplicates
	
	main_lines=pd.DataFrame()
	
	limit=1
	
	for line in duplicates:
		frame=line_frame[line_frame.Lineseries == line]
		#print frame
		
		print(str(line) + '		' + str(frame.number.sum()))
		if frame.shape[0] > limit:
			for i in range(limit):
				#print frame.iloc[i]
				main_lines=main_lines.append(frame.iloc[i])
		else:
			for i in range(frame.shape[0]):
				#print frame.iloc[i]
				main_lines=main_lines.append(frame.iloc[i])
			
	print main_lines
	
	#main_lines.to_pickle('primary_line_types.pkl')
	
	raise
	
	duplicates=[]
	for line in towers.TOWER_CONS:
		if line not in duplicates:
			duplicates += [line]
	print duplicates
	
	
	raise

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


