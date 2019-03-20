import math as m
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

## parameters are being changed to reflect new cables

def round_sig(x,sig):
    return round(x,sig-int(m.floor(m.log10(abs(x))))-1)

class cable:
	def __init__(self, name,area,R_25,a_r,I,T_lim):
		self.name = name
		self.area = area   # mm^2
		self.d = m.sqrt(4*area/m.pi)/1000	#conductor diameter/ m
		self.R_25 = R_25   # ohms/m
		self.a_r = a_r     #coefficient for resistivity wrt temp
		self.I=I		
		self.T_lim = T_lim     #rated current A

	def get_name(self):
            return self.name()


def Tc_calc(T_ambient,v,day,cable):
    ### Parameters of the line, wind, and solar conditions

    e=0.5 #emissivity

    alpha=0.5 #solar absoprtivity
        
    Ta=T_ambient

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

    lat=55.3781    # degrees North   55.3781 is latitude for UK

    He= 20  #Line elevation /m

    Zl=90 #azimuth of line

    time=11  #hour of day, only affects angle of the sun for solar radiation calculation

    conditions= 'clear'   #conditions can be clear or industrial

    Rad=m.pi/180



#######Convection loss calculation   

    Re=d*row_a*v/mu_a

    qc1f_coeff=(1.01 + 1.347*(Re**0.52))*k_f

    qc2f_coeff=0.754*(Re**0.6)*k_f

    if qc1f_coeff>qc2f_coeff:
        qcf_coeff=qc1f_coeff
    else:
        qcf_coeff=qc2f_coeff

    #print('Forced convection coeff: ' + str(qcf_coeff))

    qcn_coeff=3.645*(row_a**0.5)*(d**0.75)
    #print('Natural convection coeff: ' + str(qcn_coeff))

    #Choosing greater of natural and forced convection
    if qcf_coeff > qcn_coeff:
        qc_coeff=qcf_coeff
    else:
        qc_coeff=qcn_coeff


    



######### Radiation loss calculation

    qr_coeff=(17.8*10**-8)*d*e   # W/m

    #print('qr = ' + str(qr) + ' W/m')   # W/m



######### Solar heat gain

    delta=23.46*m.sin(Rad*360*(284+day)/365)  #solar declination /degrees

    omega=15*(time-12)   # hour angle /degrees

    Hc=m.asin(m.cos(Rad*lat)*m.cos(Rad*delta)*m.cos(Rad*omega) + m.sin(Rad*lat)*m.sin(Rad*delta))/Rad

    x=(m.sin(Rad*omega))/(m.sin(Rad*lat)*m.cos(Rad*omega) - m.cos(Rad*lat)*m.tan(Rad*delta))

    if omega >= -180:
        if omega < 0:
            if x > 0:
                Y=0
            else:
                Y=180
        else:
            if x > 0:
                Y=180
            else:
                Y=360


    Zc = Y + m.atan(x)/Rad    #solar azimuth

    theta=m.acos(m.cos(Hc)*m.cos(Zc-Zl ))/Rad    #Degrees


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

    qs=alpha*m.sin(Rad*theta)*d*Qse # W/m

    Ta=Ta+273 # Conversion to Kelvin

    if qcf_coeff >= qcn_coeff :

        #  (I**2)*R_25 - (I**2)*R_25*a_r*(25+273) + qc_coeff*Ta + qr_coeff*(Ta**4) + qs  = - (I**2)*R_25*a_r*Tc + qc_coeff*Tc + qr_coeff*(Tc**4)

        C=(I**2)*R_25 - (I**2)*R_25*a_r*(25+273) + qc_coeff*Ta + qr_coeff*(Ta**4) + qs

        # (-(I**2)*R_25*a_r + qc_coeff)*Tc + qr_coeff*(Tc**4) = C

        A = qr_coeff
        B = (-(I**2)*R_25*a_r + qc_coeff)

        # A(Tc**4) + B(Tc) - C = 0

        ### Newton's method to find steady state Tc

        # intial guess
        Tc=Ta
        prev=0
        threshold=1

        while Tc - prev > threshold:

            f= A*(Tc**4) + B*(Tc) - C

            f_dash= 4*A*(Tc**3) + B

            # iteration 
            prev=Tc
            Tc = Tc - f/f_dash

    else:

        #  I**2*R_25  - I**2*R_25*a_r*(25+273) + qs + qr_coeff*Ta**4 = qcn_coeff*(Tc-Ta)**1.25 + qr_coeff*(Tc**4) - I**2*R_25*a_r*Tc

        D= I**2*R_25  - I**2*R_25*a_r*(25+273) + qs + qr_coeff*Ta**4

        C=I**2*R_25*a_r

        B=qcn_coeff

        A=qr_coeff

        # D = B(Tc-Ta)**1.25 + A*Tc**4 - C*Tc

        # f: A*Tc**4 + B*(Tc-Ta)**1.25 - C*Tc - D = 0

        # f_dash: 4*A*Tc**3 + 1.25*B*(Tc-Ta)**0.25 - C

        

        # intial guess
        Tc=Ta
        prev=0
        threshold=1

        while Tc - prev > threshold:

            f= A*Tc**4 + B*(Tc-Ta)**1.25 - C*Tc - D 
            f_dash = 4*A*Tc**3 + 1.25*B*(Tc-Ta)**0.25 - C

            prev=Tc
            Tc=Tc - f/f_dash



    Tc=Tc-273   # degrees C
    return Tc

    

def main():

    day=180     #day of the year
                                   
    Tc=0
    
   #instantiating cable types
    hv_new=cable(name='gap_style_new',area=672,R_25=0.0478*10**-3 ,a_r=0.00403 ,I=1866,T_lim=150)

    hv_old=cable(name='acsr_hv_old',area=400,R_25=0.08*10**-3 ,a_r=0.00403 ,I=800,T_lim=75)

    lv_old=cable(name='acsr_hv_old',area=175,R_25=0.25*10**-3 ,a_r=0.00403 ,I=450,T_lim=75)

    cables=[hv_new,hv_old,lv_old]

    patches=[]

    for cable_type in cables:

        Ta_axis=[]
        v_values=[]
        Tc_values=[]
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
    plt.legend(handles=patches)
    plt.show()
        


main()
