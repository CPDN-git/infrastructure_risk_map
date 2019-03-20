## plot grid based on centroid of measurement

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.geometry import LineString
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import random

class plottable_frame():
	def __init__(self,frame=[],label='',colour='',alpha=0,size=1):
		self.frame=frame
		self.label=label
		self.colour=colour
		self.alpha=alpha
		self.size=size

	def get_frame(self):
		return self.frame

	def get_label(self):
		return self.label

	def get_colour(self):
		return self.colour

	def get_alpha(self):
		return self.alpha

	def get_size(self):
		return self.size

##################################################################
# plotting line grid, line by line

def gen_line_grid(xmin,ymin,xmax,ymax,width,height):

	
	rows = int(np.ceil((ymax-ymin)/width))
	cols = int(np.ceil((xmax-xmin)/height))


	xmin-=width/2
	ymin-=height/2
	xmax+=width/2
	ymax+=height/2

	x=xmin
	y=ymin
	strings=[]

	## vertical lines

	for i in range(cols+1):
		strings.append(LineString([Point(x,ymin),Point(x,ymax)]))
		x+=width

	x=xmin
	## horizontal lines
	for i in range(rows+1):
		strings.append(LineString([Point(xmin,y),Point(xmax,y)]))
		y+=height

	line_grid=gpd.GeoDataFrame({'geometry':strings})
	line_grid.crs = {'init' : 'epsg:4326'}

	return line_grid	


###################################################################
#  Function to plot square grid

def gen_grid_v3(xmin,ymin,xmax,ymax,width,height):


	rows = int(np.ceil((ymax-ymin)/width))
	cols = int(np.ceil((xmax-xmin)/height))


	x=xmin
	y=ymin
	polygons = []	
	colour_vals=[]
	infra_weight=[]
	for i in range(rows+1):
		x=xmin
		if i>0:
			y+=height
		for j in range(cols+1):
			polygons.append(Polygon([(x-(width/2),y+(height/2)), (x+(width/2),y+(height/2)), (x+(width/2),y-(height/2)), (x-(width/2), y-(height/2))]))
			#colour_vals+=[random.randint(0,10)]
			#infra_weight+=[0]
			x+=width
			

	#grid = gpd.GeoDataFrame({'geometry':polygons,'vals':colour_vals,'infrastructure_weight':infra_weight})
	grid = gpd.GeoDataFrame({'geometry':polygons})
	grid.crs = {'init' : 'epsg:4326'}

	return grid
		
		
	

	## generating the grid shapefile
	#points = gpd.read_file(input_file)
	#xmin,ymin,xmax,ymax =  df.total_bounds
	


#	width =  (xmax-xmin)/col_no
#	height = (ymax-ymin)/row_no
#	rows = int(np.ceil(row_no))
#	cols = int(np.ceil(col_no))
#	Xleftrigin = xmin - (width/2)
#	Xrightrigin = xmin + (width/2)
#	Ytoprigin = ymax + (height/2)
#	Ybottomrigin = ymax- (height/2)
#	polygons = []
#	colour_vals=[]
#	for i in range(cols):
#		Ytop = Ytoprigin 
#		Ybottom =Ybottomrigin
#		XleftOrigin = Xleftrigin + width*i
#		XrightOrigin = Xrightrigin + width*i
#		for j in range(rows):
#			polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin,Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
#			colour_vals+=[rows*i+j]
#			Ytop = Ytop - height
#			Ybottom = Ybottom - height


#	grid = gpd.GeoDataFrame({'geometry':polygons,'vals':colour_vals})
#	grid.crs = df.crs
	#grid.to_crs(points)
	#grid.to_file("grid.shp")
#	return grid

def run():
	britain = gpd.read_file('/home/users/jburns59/NatGridShapes/Britain/df_dno.shp')
	
	grid=gen_grid_v3(britain,xmin=-5,ymin=50.125,xmax=1.75,ymax=54.875,width=0.25,height=0.25)
	britain=britain.to_crs(grid.crs)
	#grid=gpd.read_file("grid.shp")
	#OHLs=gpd.read_file('/home/users/jburns59/NatGridShapes/OHLs/

	f, ax=plt.subplots(1)
	britain.plot(axes=ax,alpha=0.3,color= 'gold')
	grid.plot(axes=ax,column='vals',cmap='flag',alpha=0.3)
	
	plt.show()


### main
# '/home/users/jburns59/NatGridShapes/Britain/df_dno.shp'

if __name__ == "__main__":
	run()




