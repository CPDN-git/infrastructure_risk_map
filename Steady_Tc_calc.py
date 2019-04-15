import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

## parameters are being changed to reflect new cables

def round_sig(x,sig):
    return round(x,sig-int(m.floor(m.log10(abs(x))))-1)

class cable():
	def __init__(self, name,area,R_25,a_r,I,T_lim):
		self.name = name
		self.area = area   # mm^2
		self.d = m.sqrt(4*area/m.pi)/1000	#conductor diameter/ m
		self.R_25 = R_25   # ohms/m
		self.a_r = a_r     #coefficient for resistivity wrt temp
		self.I=I		   #rated current A
		self.T_lim = T_lim     #rated temperature degrees C

	def get_name(self):
            return self.name
            
def surface_temp(Ta,Tc,Tav,R_avg,I,d_0,d_core,k_th=2): ## k_th = 2 reccommended by CIGRE 207 for ACSR
	
	Ts = Tc - ((I**2)*R_avg/(2*m.pi*k_th))*(0.5 - ((d_core**2)/((d_0**2) - (d_core**2)))*(np.log(d_0/d_core)))
	
	return Ts
	
def average_temp(Ts,Tc):
	
	T_average=(Ts+Tc)/2
	
	return T_average
	
#def average_resistance(Tav,a_r):
	
	
	          
def cable_current_reduc_prob(Current, threshold):
	low = Current[Current < threshold]
	#print('currents below rated : ' + str(low))
	count= float(low.size)
	#print('number of currents below rated : ' + str(count))
	
	total = float(Current.size)
	#print('total number of currents : ' + str(total))
	
	#print(str(count)+'/'+str(total)+'='+str(count/total))	
	P=float(count/total)
	return P


def Tc_calc(Ta_data,v_data,cable,lat_index,long_index,lats):
	
	


#### deal with super-negative values that represent lack of data in Tmax files

#### when checking single grid cells, this will pop out straight away if
#### the cell doesn't have a temperature reading

	Ta=Ta_data[:,:,lat_index,long_index]

	
	v=v_data[:,:,lat_index,long_index]


	if np.any(Ta < -1):   ##kelvin
		return Ta

### Parameters of the line, wind, and solar conditions

	e=0.5 #emissivity

	alpha=0.5 #solar absoprtivity

	Area=cable.area # mm^2
	

	d= cable.d #conductor diameter/ m
	
	
	R_25=cable.R_25 # ohms/m
	
	
	a_r=cable.a_r  #coefficient for resistivity wrt temp
	
	I=cable.I #rated current A
	

# print('Ambient temperature = ' + str(Ta) + ' C')


#v=4.42422 #wind speed m/s

# print('Wind speed = ' + str(v) + ' m/s')


####### Air stats

	row_a=1.225 #kg/m^3

	mu_a=1.81*10**-5 #kg/m s

	k_f=0.02699 #W/m degree C thermal conductivity of air

	####### Conditions of line

	k_angle=1

	lat=lats[lat_index]    # degrees North   55.3781 is latitude for UK

	He= 20  #Line elevation /m

	Zl=90 #azimuth of line

	time=12  #hour of day, only affects angle of the sun for solar radiation calculation

	conditions= 'clear'   #conditions can be clear or industrial

	Rad=m.pi/180



#######Convection loss calculation   

	Re=d*row_a*v/mu_a

	#print Re

	qc1f_coeff=(1.01 + 1.347*(Re**0.52))*k_f
	
	#print qc1f_coeff

	qc2f_coeff=0.754*(Re**0.6)*k_f
	
	#print qc2f_coeff
	
	qcf_coeff=qc1f_coeff
	
	idx=np.where(qc1f_coeff < qc2f_coeff)
	#print idx
	qcf_coeff[idx]=qc2f_coeff[idx]
	#print qcf_coeff

	#if qc1f_coeff>qc2f_coeff:
	#	qcf_coeff=qc1f_coeff

	#else:
	#	qcf_coeff=qc2f_coeff
	#	
		


#print('Forced convection coeff: ' + str(qcf_coeff))

	#
#print('Natural convection coeff: ' + str(qcn_coeff))






######### Radiation loss calculation

	qr_coeff=(17.8*10**-8)*d*e   # W/m

#print('qr = ' + str(qr) + ' W/m')   # W/m



######### Solar heat gain
	
	days=np.linspace(0,v.shape[1]-1,num=v.shape[1])
	days=np.asarray(days)

	
	delta=23.46*np.sin(Rad*360*(284+days)/365)  #solar declination /degrees
	


	omega=15*(time-12)   # hour angle /degrees
	
	#print omega

	Hc=np.arcsin(np.cos(Rad*lat)*np.cos(Rad*delta)*np.cos(Rad*omega) + np.sin(Rad*lat)*np.sin(Rad*delta))/Rad

	
	
	x=(np.sin(Rad*omega))/(np.sin(Rad*lat)*np.cos(Rad*omega) - np.cos(Rad*lat)*np.tan(Rad*delta))

	#print x
	
	Y=np.full(shape=x.shape,fill_value=180)
	
	
	if omega >= -180:
		if omega < 0:
			idx=np.where(x > 0)
			Y[idx]=0
		else:
			idx=np.where(x<0)
			Y[idx]=360
			
	#print Y
	
#	if omega >= -180:
#		if omega < 0:
#			if x > 0:
#				Y=0
#			else:
#				Y=180
#		else:
#			if x > 0:
#				Y=180
#			else:
#				Y=360


	Zc = Y + np.arctan(x)/Rad    #solar azimuth

	theta=np.arccos(np.cos(Hc)*np.cos(Zc-Zl ))/Rad    #Degrees


	if conditions.lower() == 'clear':

		A=-42.2391
		B=63.8044
		C=-1.922
		D=3.46921*10**-2
		E=-3.61118*10**-4
		F=1.94318*10**-6
		G=-4.07608*10**-9

	elif conditions.lower() == 'industrial':
		A=53.1821
		B=14.2110
		C=6.6138*10**-1
		D=-3.1658*10**-2
		E=5.4654*10**-4
		F=-4.3446*10**-6
		G=1.3236*10**-8

########Solar heat flux intensity

	Qs = A + B*Hc + C*Hc**2 + D*Hc**3 + E*Hc**4 + F*Hc**5 + G*Hc**6   #W/m^2
	Qs = 0.1*10000

	A=1
	B=1.148*10**-4
	C=-1.108*10**-8

	K_sol= A + B*He + C*He**2

	#Heat flux intensity corrected for solar flux altitude

	Qse=K_sol*Qs

	qs=alpha*np.sin(Rad*theta)*d*Qse # W/m

	#Ta=Ta+273 # Conversion to Kelvin


	######## Forced convection

	#  (I**2)*R_25 - (I**2)*R_25*a_r*(25+273) + qc_coeff*Ta + qr_coeff*(Ta**4) + qs  = - (I**2)*R_25*a_r*Tc + qc_coeff*Tc + qr_coeff*(Tc**4)

	C=(I**2)*R_25 - (I**2)*R_25*a_r*(25+273) + qcf_coeff*Ta + qr_coeff*(Ta**4) + qs

	# (-(I**2)*R_25*a_r + qc_coeff)*Tc + qr_coeff*(Tc**4) = C

	A = qr_coeff
	B = (-(I**2)*R_25*a_r + qcf_coeff)

	# A(Tc**4) + B(Tc) - C = 0

	### Newton's method to find steady state Tc

	# intial guess
	Tc=Ta
	prev=np.zeros(Tc.shape)
	threshold=1
	
	idx= np.where(abs(Tc-prev)>threshold)
	#print idx
	#print Tc[idx]
	#print idx
	
	for i in range(20):
	#while len(idx) > 1:
	
		f= A*(Tc**4) + B*(Tc) - C
		
		f_dash= 4*A*(Tc**3) + B

		# iteration 
		prev=Tc
		Tc[idx] = Tc[idx] - f[idx]/f_dash[idx]
		#print Tc[idx]
		
		idx=np.where(abs(Tc-prev)>threshold)
		#print idx
		#print len(idx)

	#Tc_f=Tc
	#qc_f=qcf_coeff*(Tc-Ta)


	#while Tc - prev > threshold:

	#	f= A*(Tc**4) + B*(Tc) - C

	#	f_dash= 4*A*(Tc**3) + B

		# iteration 
	#	prev=Tc
	#	Tc = Tc - f/f_dash

	#	Tc_f=Tc
	#	qc_f=qcf_coeff*(Tc-Ta)
		
		

	# if v < 0.1:

############ Natural Convection

#  I**2*R_25  - I**2*R_25*a_r*(25+273) + qs + qr_coeff*Ta**4 = qcn_coeff*(Tc-Ta)**1.25 + qr_coeff*(Tc**4) - I**2*R_25*a_r*Tc

#   D= I**2*R_25  - I**2*R_25*a_r*(25+273) + qs + qr_coeff*Ta**4

#    C=I**2*R_25*a_r

#   B=qcn_coeff

#   A=qr_coeff

# D = B(Tc-Ta)**1.25 + A*Tc**4 - C*Tc

# f: A*Tc**4 + B*(Tc-Ta)**1.25 - C*Tc - D = 0

# f_dash: 4*A*Tc**3 + 1.25*B*(Tc-Ta)**0.25 - C



# intial guess
#    Tc=Ta
#    prev=0
#   threshold=1

#    while Tc - prev > threshold:

#            f= A*Tc**4 + B*(Tc-Ta)**1.25 - C*Tc - D 
#           f_dash = 4*A*Tc**3 + 1.25*B*(Tc-Ta)**0.25 - C

#            prev=Tc
#            Tc=Tc - f/f_dash

#        Tc_n=Tc


#       qc_n=qcn_coeff*(Tc-Ta)**1.25
#      print(str(v)+' m/s')
#     print(str(Tc))
#    print(str(Ta))
#   if qc_n > qc_f:
#      Tc=Tc_n
# else:
#    Tc=Tc_f







	Tc=Tc-273   # degrees C
	return Tc

def Current_calc(Ta_data,v_data,cable,lat_index,long_index,lats):  
	
	#### deal with super-negative values that represent lack of data in Tmax files



	Ta=Ta_data[:,150:270,lat_index,long_index]

	
	v=v_data[:,150:270,lat_index,long_index]

#### when checking single grid cells, this will pop out straight away if
#### the cell doesn't have a temperature reading

	if np.any(Ta < -1):   ##kelvin
		return cable.I + 1

### Parameters of the line, wind, and solar conditions

	e=0.5 #emissivity

	alpha=0.5 #solar absoprtivity

	Area=cable.area # mm^2
	

	d= cable.d #conductor diameter/ m
	
	
	R_25=cable.R_25 # ohms/m
	
	
	a_r=cable.a_r  #coefficient for resistivity wrt temp
	
	Tc=cable.T_lim #assume conductor at rated temperature degrees C
	Tc+=273.3   #Kelvin
	
	Ts=Tc
	
	Tfilm=(Ta+Ts)/2
	
	R_Tc=R_25*(1+a_r*(Tc-(25+273)))
	

# print('Ambient temperature = ' + str(Ta) + ' C')


#v=4.42422 #wind speed m/s

# print('Wind speed = ' + str(v) + ' m/s')


####### Air stats

	row_a=1.225 #kg/m^3

	mu_a=1.81*10**-5 #kg/m s

	k_f=0.02699 #W/m degree C thermal conductivity of air at Tfilm

	####### Conditions of line
	
	phi= 0  ## angle between wind and line, 0 < phi < m.pi/2
	
	k_angle=1.194 - np.cos(phi) + 0.194*np.cos(2*phi) + 0.368*np.sin(2*phi)
	
	lat=lats[lat_index]    # degrees North   55.3781 is latitude for UK

	He= 20  #Line elevation /m

	Zl=90 #azimuth of line

	time=12  #hour of day, only affects angle of the sun for solar radiation calculation

	conditions= 'clear'   #conditions can be clear or industrial

	Rad=m.pi/180



#######Convection loss calculation   

	Re=d*row_a*v/mu_a

	qc1f=k_angle*(1.01 + 1.347*(Re**0.52))*k_f*(Tc-Ta)

	qc2f=k_angle*0.754*(Re**0.6)*k_f*(Tc-Ta)
	
	qcn=3.645*(row_a**0.5)*(d**0.75)*(Tc-Ta)**1.25
	
	qc=qc1f
	
	#comparing qc1f to qc2f, choosing the higher one
	
	idx=np.where(qc < qc2f)
	
	qc[idx]=qc2f[idx]
	
	#same again for natural convection
	
	idx=np.where(qc<qcn)
	
	qc[idx]=qcn[idx]
	

######### Radiation loss calculation

	qr=(17.8*10**-8)*d*e*((Tc**4)-(Ta**4))   # W/m

#print('qr = ' + str(qr) + ' W/m')   # W/m



######### Solar heat gain
	
	days=np.linspace(0,v.shape[1]-1,num=v.shape[1])
	days=np.asarray(days)

	
	delta=23.46*np.sin(Rad*360*(284+days)/365)  #solar declination /degrees
	

	# time always equals 12 for our data

	omega=15*(time-12)   # hour angle /degrees
	
	#print omega

	Hc=np.arcsin(np.cos(Rad*lat)*np.cos(Rad*delta)*np.cos(Rad*omega) + np.sin(Rad*lat)*np.sin(Rad*delta))/Rad

	
	
	x=(np.sin(Rad*omega))/(np.sin(Rad*lat)*np.cos(Rad*omega) - np.cos(Rad*lat)*np.tan(Rad*delta))
	
	Y=np.full(shape=x.shape,fill_value=180)
	
	
	if omega >= -180:
		if omega < 0:
			idx=np.where(x > 0)
			Y[idx]=0
		else:
			idx=np.where(x<0)
			Y[idx]=360


	Zc = Y + np.arctan(x)/Rad    #solar azimuth

	theta=np.arccos(np.cos(Hc)*np.cos(Zc-Zl ))/Rad    #Degrees


	if conditions.lower() == 'clear':

		A=-42.2391
		B=63.8044
		C=-1.922
		D=3.46921*10**-2
		E=-3.61118*10**-4
		F=1.94318*10**-6
		G=-4.07608*10**-9

	elif conditions.lower() == 'industrial':
		A=53.1821
		B=14.2110
		C=6.6138*10**-1
		D=-3.1658*10**-2
		E=5.4654*10**-4
		F=-4.3446*10**-6
		G=1.3236*10**-8

########Solar heat flux intensity

	Qs = A + B*Hc + C*Hc**2 + D*Hc**3 + E*Hc**4 + F*Hc**5 + G*Hc**6   #W/m^2
	
	#assuming solar insolation of 1000 W/m**2
	Qs = 1000

	A=1
	B=1.148*10**-4
	C=-1.108*10**-8

	K_sol= A + B*He + C*He**2

	#Heat flux intensity corrected for solar flux altitude

	Qse=K_sol*Qs

	qs=alpha*np.sin(Rad*theta)*d*Qse # W/m
	


	# (qc + qr - qs)/R_Tc = I**2
	
	I=np.empty(shape=v.shape)
	
	for i in range(len(qs)):
		I[:,i]=np.sqrt((qc[:,i] + qr[:,i] - qs[i])/R_Tc)
		

	return I
 

def main():

    day=180     #day of the year
                                   
    Tc=0
    
   #instantiating cable types
    hv_new=cable(name='gap_style_new',area=672,R_25=0.0478*10**-3 ,a_r=0.00403 ,I=1866,T_lim=150)

    hv_old=cable(name='acsr_hv_old',area=400,R_25=0.08*10**-3 ,a_r=0.00403 ,I=800,T_lim=75)

    lv_old=cable(name='acsr_hv_old',area=175,R_25=0.25*10**-3 ,a_r=0.00403 ,I=450,T_lim=75)

    cables=[hv_new,hv_old,lv_old]

    patches=[]
    
    lats=np.linspace(50.125,54.875,num=20)
    
    

    

    for cable_type in cables:

        Ta_axis=[]
        v_values=[]
        Tc_values=[]
        raise
        for Ta in list(range(-15,50)): 
            for v in  reversed(list(range(0,3000))):
                v=v/1000
                prev=Tc
                Tc=Tc_calc(Ta,v,day,cable_type)

                if Tc > cable_type.T_lim:  # Rated temperature calculated using this script and jpower info
        
                    Ta_axis=Ta_axis + [Ta]
                    v_values=v_values + [v]
                    Tc_values=Tc_values + [Tc]
                    
                    break
                
        if cable_type==hv_new:
            col='blue'
            plt.plot(v_values,Ta_axis,color=col)
            patch= mpatches.Patch(color=col,label='Gap Style New')
            patches+=[patch]

            v=np.array(v_values)
            T=np.array(Ta_axis)
            hv_new_thresh=np.column_stack((T,v))
           # np.save('hv_new_threshold.npy',hv_new_thresh)

            print(hv_new_thresh)
            print(hv_new_thresh.shape)
            
        elif cable_type==hv_old:
            col='red'
            plt.plot(v_values,Ta_axis,color=col)
            patch= mpatches.Patch(color=col,label='ACSR High Voltage Old')
            patches+=[patch]

            v=np.array(v_values)
            T=np.array(Ta_axis)
            hv_old_thresh=np.column_stack((T,v))
            #np.save('hv_old_threshold.npy',hv_old_thresh)
           
            print(hv_old_thresh)
            print(hv_old_thresh.shape)
            
        elif cable_type==lv_old:
            col='green'
            plt.plot(v_values,Ta_axis,color=col)
            patch= mpatches.Patch(color=col,label='ACSR Low Voltage Old')
            patches+=[patch]

            v=np.array(v_values)
            T=np.array(Ta_axis)
            lv_old_thresh=np.column_stack((T,v))
           # np.save('lv_old_threshold.npy',lv_old_thresh)
           
            print(lv_old_thresh)
            print(lv_old_thresh.shape)
    plt.xticks(np.arange(0,3,0.2))
    plt.yticks(np.arange(-15,50,5))
    plt.xlabel('Wind speed m/s')
    plt.ylabel('Ambient Temperature C ')
    plt.suptitle('Thresholds at which conductors go above rated temperature, assuming running at rated current.')
    plt.legend(handles=patches)
    plt.show()
        
def upgrading_to_arrays():
	
	lat_index = 10
	long_index = 26

	#instantiating cable types
	hv_new=cable(name='gap_style_new',area=672,R_25=0.0478*10**-3 ,a_r=0.00403 ,I=1866,T_lim=150)
	hv_old=cable(name='acsr_hv_old',area=400,R_25=0.08*10**-3 ,a_r=0.00403 ,I=800,T_lim=75)
	lv_old=cable(name='acsr_lv_old',area=175,R_25=0.25*10**-3 ,a_r=0.00403 ,I=450,T_lim=75)

	cables=[hv_new,hv_old,lv_old]
	


	dataTmax=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_648_data_item3236_daily_maximum_full_field.npy')
	dataWind=np.load('/home/users/jburns59/cpdn_analysis/examples/batch_648_data_item3249_daily_mean_full_field.npy')
	
	dataTmax=np.squeeze(dataTmax)
	dataWind=np.squeeze(dataWind)
	
	
	#y=np.delete(dataWindHist,664,axis=0)
	
	#print dataTmaxHist.shape
	#print dataWindHist.shape	
	
	#dataWindHist=np.resize(dataWindHist,dataTmaxHist.shape)
	
	lats=np.linspace(50.125,54.875,num=20)
	
	for cable_type in cables:

		print('Cable : ' + cable_type.get_name())
		I=Current_calc(Ta_data=dataTmax,v_data=dataWind,cable=cable_type,lat_index=lat_index,long_index=long_index,lats=lats)
		
		
		## maybe problem is coming from use of Tc instead of Tfilm for certain values such as air density
		## mainly from wind orientation actually
		
		P=cable_current_reduc_prob(I,cable_type.I)
	
	
		print P
	
	
	

#testing_convection()
if __name__ == '__main__':
	upgrading_to_arrays()
	#testing()
	#main()
