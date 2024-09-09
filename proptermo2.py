# -*- coding: cp1252 -*-
from numpy import array,dot,r_,log,size,zeros,log10
from pylab import plot, grid, show, legend,figure
from pylab import title, xlabel, ylabel,subplot,text
from constants import *

def calcula_props(T,P=100.,Po=100.):
    Coef_eqs = array((
                [1,T,T**2,T**3,T**4,0,0],
                [1.,.5*T, 1./3*T**2, .25*T**3, 1./5*T**4, 1./T,0],
                [log(T),T,0.5*(T**2),(1.0/3)*T**3,0.25*T**4,0,1]
                     ))

    RuP = Ru*log(P/Po)

    if 300.<= T <= 1000.:
        M = MMols
        subs = subst
        cpa = dot(Coef_low,Coef_eqs[0,:])
        ha  = dot(Coef_low,Coef_eqs[1,:])
        soa = dot(Coef_low,Coef_eqs[2,:])

    if 1000.< T <=3000.:
        M = MMols[:10]
        subs = subst[:10]
        cpa = dot(Coef_high,Coef_eqs[0,:])
        ha  = dot(Coef_high,Coef_eqs[1,:])
        soa = dot(Coef_high,Coef_eqs[2,:])

    cp,h,s = cpa*(Ru),ha*(Ru*T),soa*Ru - RuP
    
    cpd = dict(zip(subs,cp) )
    hd = dict(zip(subs,h) )
    sd = dict(zip(subs,s) )
    Md = dict(zip(subs,M) )

    return cpd,hd,sd,Md

def calcula_constEq(T):
    Coef_eqs = array(( [log(T/1000.), 1./T, 1., 1.*T, T**2] ))
    K = 10.**dot(Coef_K,Coef_eqs)

    Kd = dict(zip(range(1,7),K) )

    return Kd

if __name__ == '__main__':

    Tlist = r_[300.:500.:50.,500:3000.:300]
    P,Po = 1000.,100.
    
    K,Ki = [],[None]
    
    for T in Tlist:
        print 'T:',T,'P:',P,'Po:',Po
        if T >= 300:
            KT = calcula_constEq(T)
            print KT
            K.append(KT)
       
        cp,h,s,M = calcula_props(T,P)
        
        print '%s:\t\t%s\t\t%s\t\t%s' %('subst', 'cp', 'h', 's')
        for sub in subst:
            try:
                if sub == 'C14.4H24.9':
                    print '%s: %8.3f\t%12.3f\t%12.3f' %(sub, cp[sub], h[sub], s[sub])
                else:
                    print '%s:\t%12.3f\t%12.3f\t%12.3f' %(sub, cp[sub], h[sub], s[sub])
            except KeyError:
                pass
        print
    
    lenT = range(size(Tlist))
    props = [calcula_props(T,P) for T in Tlist]
    Ki.extend([[K[j][i] for j in lenT] for i in range(1,7)])

    prop2 = zeros((3,5,size(Tlist)) )
    for j in [0,1,2]: #cp,h,s
        for k,sub in enumerate(['CO2','H2O','N2','O2','OH']):
            prop2[j,k,:] = ([props[i][j][sub] for i,T in enumerate(Tlist)])

    print prop2
    
    figure(1)
    lab = {0:'Cp (kJ/kmolK)',1:'h (MJ/kmol)',2:'s (kJ/kmolK'} # labels em y
    for j in [0,1,2]: # cp,h,s
        prCO2,prH2O,prN2,prO2,prOH = prop2[j,:,:]
        if j == 1: # se for h...
            prCO2,prH2O,prN2,prO2,prOH = (prCO2/1000.,prH2O/1000.,
                                                prN2/1000.,prO2/1000.,prOH/1000.)
        subplot(1,4,j+1)
        
        plot(Tlist/1000.,prCO2,'g',Tlist/1000.,prH2O,'b',
             Tlist/1000.,prN2,'r',Tlist/1000.,prO2,'y',
             Tlist/1000.,prOH,'m' )
        
        legend(('CO2','H2O','N2','O2','OH'))
        title(lab[j])
        if j == 2:
            xp,yp = 1.,180.
            text(xp,yp,'P = %.1f kPa'%P)
        xlabel('Tx1000 (K)')       
        grid()

    subplot(1,4,4)
    for i in range(1,7):
        plot(1000./Tlist,log10(Ki[i]))
    grid()
    xlabel('1/Tx0.001 (1/K)')
##    ylabel('log10 Ki')
    legend(('K1','K2','K3','K4','K5','K6'))
    title(u'Ctes de Eq (log10 Ki)')
    
    show()
                                        
    
