
# -*- coding: cp1252 -*-
from scipy import *
from numpy.linalg import *
from math import log as ln
from constants import *
from robustNR_args import robustNewton
import proptermo2 as p2
import sys
  
def calcula_prods(alfa,beta,gama,delta,phi,T,P,
                  Po= 100.,xtol=1.e-08, jacob = 1):
    """ Para o combustível composto de C alfa H beta O gama e N delta e razão
de equivalência phi, esta função calcula o equilíbrio dos dez produtos de
combustão mais comuns, à temperatura e pressão T (de 300 a 3000K) e P (kPa):
		CO2, H2O, N2, O2, CO, H2, H, O, OH e NO,
usando 6 equações de equilíbrio, com jacobiano algébrico, retornando as
frações molares e o número total de kmoles por kmol de combustível, além
das propriedades *molares* da mistura, como gá perfeito: calor específico, 
entalpia, entropia, volume específico molares, massa molecular, bem como
o vetor dos resíduos (que devem ser próximos de zero) do sistema de equações
cujas raízes devem ser obtidas por um esquema de Newton-Raphson modificado.
NOTA IMPORTANTE: Se for utilizado jacobiano *numérico* - argumento default - e
se phi for *igual* a 1.0 e só nesse caso, o jacobiano resultará *singular*
para valores de T abaixo de 900K...
    """

    ast = alfa + beta/4. - gama/2. # relação ar-combustível molar estequiométrica

    PPo = P/Po      # relação de pressóes, onde Po é a pressão de referência (kPa)
    
    K = p2.calcula_constEq(T) # constantes de equilíbrio em função da temperatura T
    
    # definições por conveniência:
    LK1 = ln(K[1]*K[1]/PPo)
    LK2 = ln(K[2]*K[2]/PPo)
    LK3 = ln(K[3]*K[3])
    LK4 = ln(K[4]*K[4])
    LK5 = ln(K[5]*K[5]*PPo)
    LK6 = ln(K[6]*K[6]*PPo)
    astphi = 2*ast/phi
        
    def equations(n,args=None):
        """
        Monta o sistema de equações, cujas raízes
        devem ser obtidas - n é o vetor número de moles
        """
        
        n1,n2,n3,n4,n5,n6,n7,n8,n9,n10 = n
        
        N = n.sum()
        
        eqs = array((
             (n1 + n5) - alfa,
             (2*n2 + 2*n6 + n7 + n9) - beta,
             (2*n1 + n2 + 2*n4 + n5 + n8 + n9 + n10) - (gama + asphi),
             (2*n3 + n10) - (delta + 3.76*asphi),
             2*ln(n7) - ln(n6) - ln(N) - LK1,
             2*ln(n8) - ln(n4) - ln(N) - LK2,
             2*ln(n9) - ln(n4) - ln(n6) - LK3,
             2*ln(n10) - ln(n4) - ln(n3) - LK4,
             2*ln(n2) - ln(n4) - 2*ln(n6) + ln(N) - LK5,
             2*ln(n1) - ln(n4) - 2*ln(n5) + ln(N) - LK6
                     ))
                
        return eqs

    def jacobiano(n,args=None):
        """
        Calcula o jacobiano do sistema de equações -
        As derivadas são calculadas ao longo das linhas.
        """
        n1,n2,n3,n4,n5,n6,n7,n8,n9,n10 = n

        N = n.sum()
        
        jac = array((
            [1,0,0,0,1,0,0,0,0,0],
            [0,2,0,0,0,2,1,0,1,0],
            [2,1,0,2,1,0,0,1,1,1],
            [0,0,2,0,0,0,0,0,0,1],
            [0,0,0,0,0,-1./n6,2./n7,0,0,0],
            [0,0,0,-1./n4,0,0,0,2./n8,0,0],
            [0,0,0,-1./n4,0,-1./n6,0,0,2./n9,0],
            [0,0,-1./n3,-1./n4,0,0,0,0,0,2./n10],
            [0,2./n2,0,-1./n4,0,-2./n6,0,0,0,0],
            [2./n1,0,0,-1./n4,-2./n5,0,0,0,0,0]
                    ))

        jac += reshape(
            array([0,0,0,0,-1./N,-1./N,0,0,1./N,1./N]),(10,1)
                       )
        return jac

    def calcula_props_mist(fracs,T,P):
        """
        Através dessa função são determinadas as propriedades da mistura de
        produtos de combustão, como gás perfeito, a T e P, conhecidas as
        respectivas frações molares do equilíbrio (fracs).
        """
        cp,h,s,Ms = p2.calcula_props(T,P)

        hmist = (h['CO2']*fracs['CO2'] + h['H2O']*fracs['H2O'] +
                   h['N2']*fracs['N2'] + h['O2']*fracs['O2'] +
                   h['CO']*fracs['CO'] + h['H2']*fracs['H2'] +
                   h['H']*fracs['H'] + h['O']*fracs['O'] +
                   h['OH']*fracs['OH'] + h['NO']*fracs['NO'])

        Mmist = (Ms['CO2']*fracs['CO2'] + Ms['H2O']*fracs['H2O'] +
                 Ms['N2']*fracs['N2'] + Ms['O2']*fracs['O2'] +
                 Ms['CO']*fracs['CO'] + Ms['H2']*fracs['H2'] +
                 Ms['H']*fracs['H'] + Ms['O']*fracs['O'] +
                 Ms['OH']*fracs['OH'] + Ms['NO']*fracs['NO'])

        smist = (s['CO2']*fracs['CO2'] + s['H2O']*fracs['H2O'] +
                   s['N2']*fracs['N2'] + s['O2']*fracs['O2'] +
                   s['CO']*fracs['CO'] + s['H2']*fracs['H2'] +
                   s['H']*fracs['H'] + s['O']*fracs['O'] +
                   s['OH']*fracs['OH'] + s['NO']*fracs['NO']) -\
                Ru*(fracs['CO2']*ln(fracs['CO2']) + fracs['H2O']*ln(fracs['H2O']) +
                   fracs['N2']*ln(fracs['N2']) + fracs['O2']*ln(fracs['O2']) +
                   fracs['CO']*ln(fracs['CO']) + fracs['H2']*ln(fracs['H2']) +
                   fracs['H']*ln(fracs['H']) + fracs['O']*ln(fracs['O']) +
                   fracs['OH']*ln(fracs['OH']) + fracs['NO']*ln(fracs['NO'])  ) 

        cpmist = (cp['CO2']*fracs['CO2'] + cp['H2O']*fracs['H2O'] +
                   cp['N2']*fracs['N2'] + cp['O2']*fracs['O2'] +
                   cp['CO']*fracs['CO'] + cp['H2']*fracs['H2'] +
                   cp['H']*fracs['H'] + cp['O']*fracs['O'] +
                   cp['OH']*fracs['OH'] + cp['NO']*fracs['NO'])

        vmist = Ru*T/P

        return cpmist,hmist,smist,vmist,Mmist
         
    #************************
    # corpo da função calcula_prods:
    # condições iniciais:
    
    n_init = ones((10))

    if jacob is not None:
##        print 'Com Jacobiano...'
        jacob = jacobiano
        
    try:
        n,ite,F = robustNewton(equations,n_init,
                        jacob = jacob,
                        xtol = xtol,args=None)

##        print ite   
    except LinAlgError:
        print 'T: ',T,'P: ',P,'phi: ',phi
        print "Jacobiano singular ou não fornecido!"
        raw_input('Aperte RETURN para sair...')
        sys.exit()
                   
    N = n.sum()
    y = n/N
    yd = dict(zip(subst,y) )

    cp,h,s,v,M = calcula_props_mist(yd,T,P)
    
    return yd,N,cp,h,s,v,M,ite,F

if __name__ == '__main__':
    T = 300.
    P = 100.
    phi = 1.0000
    jacob = 1
    
    if jacob is None:
        print 'Sem Jacobiano...'
    else:
        print 'Com Jacobiano...'
        
    fracs,N,cp,h,s,v,M,ite,F = calcula_prods(7.,17.,0.,0.,phi,T,P, jacob = jacob)

    
    print ite
    print 'T: ',T,'P: ',P,'phi: ',phi
    print 'ntotal: ', N,'\nfracs:\n',fracs
    print 'propriedades da mistura:\ncp: %f kJ/kgK, h: %f kJ/kg, s: %f kJ/kgK, v:%f m3/kg, M: %f kg/kmol'%(
                                                        cp/M,h/M,s/M,v/M,M)
    print 'resíduos:\n',F

    
