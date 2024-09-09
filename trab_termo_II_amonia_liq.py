# -*- coding: cp1252 -*-

# Grupo 10.1: Termoman�acos - O Retorno -
# Autores: R�binson Erazo e Agmar Pereira
# RAs respectivos: 200711491 e 200712331
# Contato: robinson.erazo@hotmail.com e agmar_filho@hotmail.com

"""Programa desenvolvido para resolver o exerc�cio 12.81 do livro Fundamentos
   da Termodin�mica, 4�ed., de Van Wylen et alli. Abaixo o enunciado do pro-
   blema:

   12.81 Escreva um programa de computador para resolver uma generaliza��o do
   prob. 12.37. Utilize a rela��o de compress�o, a efici�ncia isoentr�pica do
   compressor e a temperatura adiab�tica de chama como vari�veis de entrada
   do programa.

   12.37 Um estudo est� sendo realizado para avaliar se a am�nia l�quida � um
   combust�vel adequado para uma turbina a g�s. Considere os processos de com-
   press�o e combust�o deste equipamento.

   a. Ar entra no compressor a 100kPa e 25�C. Este � comprimido at� 1600kPa e
   a efici�ncia isoentr�pica � 87%. Determine a temperatura de descarga do ar
   e o trabalho consumido por kmol de ar.

   b. Dois kmoles de am�nia l�quida a 25�C e x vezes do ar te�rico, proveni-
   entes do compressor, entram na c�mara de combust�o. Qual o valor de x se a
   temperatura adiab�tica de chama for fixada em 1600 K.

   Assim, a partir da rela��o de compress�o, da efici�ncia isoentr�pica do
   compressor e da temperatura adiab�tica de chama, o presente programa forne-
   cer� a temperatura de descarga do ar (item a.), o trabalho consumido por kmol
   de ar (item a.) e o valor de x vezes de ar te�rico (item b.)"""

from math import *
from numpy import array
from scipy.integrate import quad
import equilibriumNR as equi
import robustNR_args


"--------- FUN��ES QUE CALCULAM AS PROPRIEDADES TERMODIN�MICAS DO AR ---------"

"Constantes importantes do ar"

M_ar = 28.97 #kg/kmol
Ru = 8.13451 #kJ/kmol.K

'Oxig�nio molecular'
y_O2  = 0.21 # Fra��o Molar de O2 no ar
M_O2  = 32.000 # kg/kmol

'Nitrog�nio molecular'
y_N2  = 0.79 # Fra��o Molar de N2 no ar
M_N2  = 28.013 # kg/kmol

"C�lculo de Cp0(T) do Oxig�nio molecular."

def Cp0_O2(T): # Cp(T) � dado em kJ/kmol.K
            
    cp0 =  2.811 * pow(10, 1)  # Coeficientes do calor espec�fico do oxig�nio
    cp1 = -3.680 * pow(10,-6)  # Retirado do ap�ndice A de Ferguson, C. R.
    cp2 =  1.746 * pow(10,-5) 
    cp3 = -1.065 * pow(10,-8) 

    return cp0 + cp1*pow(T,1)  +  cp2*pow(T,2)  +  cp3*pow(T,3)    

'---------------------------------------------------------------------------'

"Calcula a antidiferencial de Cp0(T) para calcularmos integral SCp0(T)dT."

def Scp0_O2(T): # Cp(T) � dado em kJ/kmol.K
            
    cp0 =  2.811 * pow(10, 1)  # Coeficientes do calor espec�fico do oxig�nio
    cp1 = -3.680 * pow(10,-6)  # Retirado do ap�ndice A de Ferguson, C. R.
    cp2 =  1.746 * pow(10,-5) 
    cp3 = -1.065 * pow(10,-8) 

    return cp0*T + cp1*pow(T,2)/2  +  cp2*pow(T,3)/3  +  cp3*pow(T,4)/4    

'---------------------------------------------------------------------------'

"C�lculo de Cp0(T) do Nitrog�nio molecular."

def Cp0_N2(T): # Cp(T) � dado em kJ/kmol.K
            
    cp0 =  3.115 * pow(10, 1)  # Coeficientes do calor espec�fico do nitrog�nio
    cp1 = -1.357 * pow(10,-2)  # Retirado do ap�ndice A de Ferguson, C. R.
    cp2 =  2.680 * pow(10,-5) 
    cp3 = -1.168 * pow(10,-8)  

    return cp0 + cp1*pow(T,1)  +  cp2*pow(T,2)  +  cp3*pow(T,3)    

'---------------------------------------------------------------------------'

"Calcula a antidiferencial de Cp0(T) para calcularmos integral SCp0(T)dT."

def Scp0_N2(T): # Cp(T) � dado em kJ/kmol.K
            
    cp0 =  3.115 * pow(10, 1)  # Coeficientes do calor espec�fico do nitrog�nio
    cp1 = -1.357 * pow(10,-2)  # Retirado do ap�ndice A de Ferguson, C. R.
    cp2 =  2.680 * pow(10,-5) 
    cp3 = -1.168 * pow(10,-8) 

    return cp0*T + cp1*pow(T,2)/2  +  cp2*pow(T,3)/3  +  cp3*pow(T,4)/4 

'---------------------------------------------------------------------------'

"C�lculo de Cp0_ar(T)do Ar considerando g�s ideal."

def Cp0_ar(T): # dado em kJ/kmol.K 

    return y_O2 * Cp0_O2(T) + y_N2 * Cp0_N2(T)    

'---------------------------------------------------------------------------'

'Entalpia entre temperaturas T1 e T2 para ar considerado g�s perfeito'

def del_h(T1,T2): # Dado e kJ/kmol

    integral_T1_T2_O2 = Scp0_O2(T2) - Scp0_O2(T1)
    integral_T1_T2_N2 = Scp0_N2(T2) - Scp0_N2(T1)

    'h2-h1 = Scp0(T)dT'
    delta_H = y_O2 * integral_T1_T2_O2 + y_N2 * integral_T1_T2_N2 

    return delta_H

'---------------------------------------------------------------------------'

'Entropia entre temperaturas (T1,P1) e (T2,P2) para ar considerado g�s perfeito'

def del_s(T1,T2,P1,P2): # Dado em kJ/kmol.K

    delta_s = quad(lambda T: (Cp0_ar(T)/T) , T1, T2)[0] - Ru*log(P2/P1)

    return delta_s
    


"--------------- FUN��ES QUE CALCULAM OS DADOS PARA O ITEM A. --------------"

"""Determina��o da temperatura IDEAL de sa�da do ar T2, considerando um
   processo isoentr�pico do estado 1 para 2"""

def acha_T2_ideal(rel_compress): # dado em Kelvin


    def func(x,args = None): #Fun��o que reprensenta o sistema de equa��es
                             #em que s2 - s1 = 0 [isoentr�pico]
                             
        T1 = 25 + 273.15 #temperatura padr�o de 25�C em Kelvin
        P1 = 100.        #press�o padr�o em kPa
        T2 = x           #nossa inc�gnita

        'P2 � determinado a partir da rela��o de compress�o (P2/P1)'
        P2 = rel_compress*P1 

        return del_s(T1,T2,P1,P2)
    
    #resolu��o do sistema de eq's
    T2_ideal = robustNR_args.robustNewton(func,300.) 

    return T2_ideal[0] # dado em Kelvin
    
'---------------------------------------------------------------------------'


"""Esta fun��o calcula o trabalho real desenvolvido pelo compressor sobre
    o ar. Foi utilizado como conven��o que o trabalho � positivo quando �
    aplicado PELO sistema SOBRE o ambiente."""

def acha_w_real(rel_compress , efic_isoentrop): # Dado e kJ/kmol

    T1 = 25 + 273.15 # temperatura padr�o de 25�C em Kelvin
    T2_ideal = acha_T2_ideal(rel_compress) # temperatura em 2 ideal

    w_ideal = del_h(T1,T2_ideal) #trabalho ideal para temperatura de 2

    w_real = w_ideal/efic_isoentrop #trabalho real do compressor

    return w_real #kJ/kmol


'---------------------------------------------------------------------------'

"""Determina��o da temperatura REAL de sa�da do ar T2, considerando o trabalho
   real corrigido do trabalho isoentr�pico"""

def acha_T2_real(w_real): # dado em Kelvin


    def func(x,args = None): #Fun��o que reprensenta o sistema de equa��es
                             #em que h2 - h1 - w_real = 0 [processo real]
                             
        T1 = 25 + 273.15 #temperatura padr�o de 25�C em Kelvin
        T2 = x           #nossa inc�gnita
 

        return del_h(T1,T2) - w_real
    
    #resolu��o do sistema de eq's
    T2_real = robustNR_args.robustNewton(func,300.) 

    return T2_real[0] # dado em Kelvin

'---------------------------------------------------------------------------'


"""Finalmente, a fun��o que resolve o item a., com rela��o de compress�o
   e a efici�ncia isoentr�pica do compressor e como sa�da teremos a tem-
   peratura e trabalho por kmol de ar na situa��o ideal e na situa��o real"""

def resolve_item_a(rel_compress , efic_isoentrop): #dados de entrada

     'C�lculo do trabalho real no compressor e a temperatura de sa�da do ar'    
     w_real = acha_w_real(rel_compress , efic_isoentrop) # kJ/kmol
     T2_real = acha_T2_real(w_real)                      # K
     
     return { 'temperatura_T2':T2_real , 'trabalho_w': w_real} #dados de sa�da


"--------------- FUN��ES QUE CALCULAM OS DADOS PARA O ITEM B. --------------"

"""Determina��o de x vezes de ar te�rico (1/phi) para uma dada temperatura
   adiab�tica de chama na c�mara de combust�o"""

def acha_X(T2_ar,T_adiab,rel_compress): 


    def func(x,args = None): #Fun��o que reprensenta o sistema de equa��es
                             #envolvendo balan�o de massas e energia na c�mara
                             #de combust�o

        X , a_s = x #nossas inc�gnitas s�o o x vezes de ar teorico e a
                    #quantidade estequiom�trica de ar

        'Constantes �teis'
        P1 = 100. #kPa
        P2 = rel_compress * P1  #kPa                   
        T0 = 25 + 273.15 #temperatura 25�C em Kelvin

        'Massas Moleculares'
        M_ar  = 28.97 #kg/kmol
        M_NH3 = 17.031 #kg/kmol
        M_mist = equi.calcula_prods(0.,3.,0.,1.,1./X,T_adiab,P2, jacob = 1)[6] #kg/kmol
        


        def h_mist(X,T,P): #uma fun��o mais conveniente da entalpia dos produtos
                           #da combust�o

            phi = 1./X # rela��o de equival�ncia e x
            M_mist = equi.calcula_prods(0.,3.,0.,1.,phi,T,P, jacob = 1)[6] #kg/kmol
            h_mist = equi.calcula_prods(0.,3.,0.,1.,phi,T,P, jacob = 1)[3] #kJ/kg 
            h_mist = h_mist * M_mist #kJ/kmol

            return h_mist

        """A seguir as quantidades de kmoles entrando e saindo do volume de
           controle em fun��o de x e a_s"""
        n_ar = 4.76 * X * a_s #kmol
        n_NH3 = 2. #kmol
        n_mist = 2 * equi.calcula_prods(0.,3.,0.,1.,1./X,T_adiab,P2, jacob = 1)[1] #kmol

        'No caso -80800kJ/kmol � a entalpia de forma��o de NH3 l�quido'
        return array (( n_ar * del_h(T0,T2_ar) + 2 * (-80800) - n_mist * h_mist(X,T_adiab,P2),
                        M_ar * n_ar + M_NH3 * n_NH3 - M_mist * n_mist,
                        )  )#duas equa��es a duas inc�gnitas (balan�os)
    
    'resolu��o do sistema de equa��es'
    X = robustNR_args.robustNewton(func,(1.2,1.2))

    return X[0][0] 


'---------------------------------- PRINCIPAL ---------------------------------'

if __name__ == '__main__':

    print """\n
   Programa desenvolvido para resolver o exerc�cio 12.81 do livro Fundamentos
   da Termodin�mica, 4�ed., de Van Wylen et alli. Abaixo o enunciado do pro-
   blema:

   12.81 Escreva um programa de computador para resolver uma generaliza��o do
   prob. 12.37. Utilize a rela��o de compress�o, a efici�ncia isoentr�pica do
   compressor e a temperatura adiab�tica de chama como vari�veis de entrada
   do programa.

   12.37 Um estudo est� sendo realizado para avaliar se a am�nia l�quida � um
   combust�vel adequado para uma turbina a g�s. Considere os processos de com-
   press�o e combust�o deste equipamento.

   a. Ar entra no compressor a 100kPa e 25�C. Este � comprimido at� 1600kPa e
   a efici�ncia isoentr�pica � 87%. Determine a temperatura de descarga do ar
   e o trabalho consumido por kmol de ar.

   b. Dois kmoles de am�nia l�quida a 25�C e x vezes do ar te�rico, proveni-
   entes do compressor, entram na c�mara de combust�o. Qual o valor de x se a
   temperatura adiab�tica de chama for fixada em 1600 K."""
 
    print """\n
    Entre com o valor da rela��o de compress�o(press�o de sa�da
    pela de entrada (P2/P1), p.e., 16.1). Lembrando que a
    press�o de entrada � de 100kPa:\n"""

    rel_compress = float(raw_input('Rela��o de compress�o:'))

    print """\n
    Entre com o valor efici�ncia isoentr�pica do compressor (p.e., ef=0.87):\n"""

    efic_isoentrop = float(raw_input('Efici�ncia isoentr�pica:'))

    print """\n
    Entre com a temperatura adiab�tica de chama na c�mara de combust�o
    em kelvins (p.e., 1600). Este programa tem uma amplitude de tempera-
    turas aceitas de 300K at� 3000K :\n"""

    T_adiab = float(raw_input('Temperatura adiab�tica:'))

temp = resolve_item_a(rel_compress , efic_isoentrop)['temperatura_T2']
trab = resolve_item_a(rel_compress , efic_isoentrop)['trabalho_w']
x    = acha_X( temp , T_adiab , rel_compress)

print """\nA partir de uma rela��o de compress�o de %.2f , efici�ncia
isoentr�pica de %.3f e temperatura adiab�tica de %.1fK,
obteremos como resultados:""" %(rel_compress,efic_isoentrop,T_adiab)

print "\nTemperatura de sa�da do ar no compressor(K):" , temp
print "\nTrabalho real realizado pelo compressor (kJ/kmol):" , trab
print "\nValor de x vezes de ar te�rico:" , x

print "\nObrigado pela aten��o!"





        
        
