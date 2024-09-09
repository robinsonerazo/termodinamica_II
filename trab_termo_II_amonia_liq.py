# -*- coding: cp1252 -*-

# Grupo 10.1: Termomaníacos - O Retorno -
# Autores: Róbinson Erazo e Agmar Pereira
# RAs respectivos: 200711491 e 200712331
# Contato: robinson.erazo@hotmail.com e agmar_filho@hotmail.com

"""Programa desenvolvido para resolver o exercício 12.81 do livro Fundamentos
   da Termodinâmica, 4ªed., de Van Wylen et alli. Abaixo o enunciado do pro-
   blema:

   12.81 Escreva um programa de computador para resolver uma generalização do
   prob. 12.37. Utilize a relação de compressão, a eficiência isoentrópica do
   compressor e a temperatura adiabática de chama como variáveis de entrada
   do programa.

   12.37 Um estudo está sendo realizado para avaliar se a amônia líquida é um
   combustível adequado para uma turbina a gás. Considere os processos de com-
   pressão e combustão deste equipamento.

   a. Ar entra no compressor a 100kPa e 25°C. Este é comprimido até 1600kPa e
   a eficiência isoentrópica é 87%. Determine a temperatura de descarga do ar
   e o trabalho consumido por kmol de ar.

   b. Dois kmoles de amônia líquida a 25°C e x vezes do ar teórico, proveni-
   entes do compressor, entram na câmara de combustão. Qual o valor de x se a
   temperatura adiabática de chama for fixada em 1600 K.

   Assim, a partir da relação de compressão, da eficiência isoentrópica do
   compressor e da temperatura adiabática de chama, o presente programa forne-
   cerá a temperatura de descarga do ar (item a.), o trabalho consumido por kmol
   de ar (item a.) e o valor de x vezes de ar teórico (item b.)"""

from math import *
from numpy import array
from scipy.integrate import quad
import equilibriumNR as equi
import robustNR_args


"--------- FUNÇÔES QUE CALCULAM AS PROPRIEDADES TERMODINÂMICAS DO AR ---------"

"Constantes importantes do ar"

M_ar = 28.97 #kg/kmol
Ru = 8.13451 #kJ/kmol.K

'Oxigênio molecular'
y_O2  = 0.21 # Fração Molar de O2 no ar
M_O2  = 32.000 # kg/kmol

'Nitrogênio molecular'
y_N2  = 0.79 # Fração Molar de N2 no ar
M_N2  = 28.013 # kg/kmol

"Cálculo de Cp0(T) do Oxigênio molecular."

def Cp0_O2(T): # Cp(T) é dado em kJ/kmol.K
            
    cp0 =  2.811 * pow(10, 1)  # Coeficientes do calor específico do oxigênio
    cp1 = -3.680 * pow(10,-6)  # Retirado do apêndice A de Ferguson, C. R.
    cp2 =  1.746 * pow(10,-5) 
    cp3 = -1.065 * pow(10,-8) 

    return cp0 + cp1*pow(T,1)  +  cp2*pow(T,2)  +  cp3*pow(T,3)    

'---------------------------------------------------------------------------'

"Calcula a antidiferencial de Cp0(T) para calcularmos integral SCp0(T)dT."

def Scp0_O2(T): # Cp(T) é dado em kJ/kmol.K
            
    cp0 =  2.811 * pow(10, 1)  # Coeficientes do calor específico do oxigênio
    cp1 = -3.680 * pow(10,-6)  # Retirado do apêndice A de Ferguson, C. R.
    cp2 =  1.746 * pow(10,-5) 
    cp3 = -1.065 * pow(10,-8) 

    return cp0*T + cp1*pow(T,2)/2  +  cp2*pow(T,3)/3  +  cp3*pow(T,4)/4    

'---------------------------------------------------------------------------'

"Cálculo de Cp0(T) do Nitrogênio molecular."

def Cp0_N2(T): # Cp(T) é dado em kJ/kmol.K
            
    cp0 =  3.115 * pow(10, 1)  # Coeficientes do calor específico do nitrogênio
    cp1 = -1.357 * pow(10,-2)  # Retirado do apêndice A de Ferguson, C. R.
    cp2 =  2.680 * pow(10,-5) 
    cp3 = -1.168 * pow(10,-8)  

    return cp0 + cp1*pow(T,1)  +  cp2*pow(T,2)  +  cp3*pow(T,3)    

'---------------------------------------------------------------------------'

"Calcula a antidiferencial de Cp0(T) para calcularmos integral SCp0(T)dT."

def Scp0_N2(T): # Cp(T) é dado em kJ/kmol.K
            
    cp0 =  3.115 * pow(10, 1)  # Coeficientes do calor específico do nitrogênio
    cp1 = -1.357 * pow(10,-2)  # Retirado do apêndice A de Ferguson, C. R.
    cp2 =  2.680 * pow(10,-5) 
    cp3 = -1.168 * pow(10,-8) 

    return cp0*T + cp1*pow(T,2)/2  +  cp2*pow(T,3)/3  +  cp3*pow(T,4)/4 

'---------------------------------------------------------------------------'

"Cálculo de Cp0_ar(T)do Ar considerando gás ideal."

def Cp0_ar(T): # dado em kJ/kmol.K 

    return y_O2 * Cp0_O2(T) + y_N2 * Cp0_N2(T)    

'---------------------------------------------------------------------------'

'Entalpia entre temperaturas T1 e T2 para ar considerado gás perfeito'

def del_h(T1,T2): # Dado e kJ/kmol

    integral_T1_T2_O2 = Scp0_O2(T2) - Scp0_O2(T1)
    integral_T1_T2_N2 = Scp0_N2(T2) - Scp0_N2(T1)

    'h2-h1 = Scp0(T)dT'
    delta_H = y_O2 * integral_T1_T2_O2 + y_N2 * integral_T1_T2_N2 

    return delta_H

'---------------------------------------------------------------------------'

'Entropia entre temperaturas (T1,P1) e (T2,P2) para ar considerado gás perfeito'

def del_s(T1,T2,P1,P2): # Dado em kJ/kmol.K

    delta_s = quad(lambda T: (Cp0_ar(T)/T) , T1, T2)[0] - Ru*log(P2/P1)

    return delta_s
    


"--------------- FUNÇÔES QUE CALCULAM OS DADOS PARA O ITEM A. --------------"

"""Determinação da temperatura IDEAL de saída do ar T2, considerando um
   processo isoentrópico do estado 1 para 2"""

def acha_T2_ideal(rel_compress): # dado em Kelvin


    def func(x,args = None): #Função que reprensenta o sistema de equações
                             #em que s2 - s1 = 0 [isoentrópico]
                             
        T1 = 25 + 273.15 #temperatura padrão de 25°C em Kelvin
        P1 = 100.        #pressão padrão em kPa
        T2 = x           #nossa incógnita

        'P2 é determinado a partir da relação de compressão (P2/P1)'
        P2 = rel_compress*P1 

        return del_s(T1,T2,P1,P2)
    
    #resolução do sistema de eq's
    T2_ideal = robustNR_args.robustNewton(func,300.) 

    return T2_ideal[0] # dado em Kelvin
    
'---------------------------------------------------------------------------'


"""Esta função calcula o trabalho real desenvolvido pelo compressor sobre
    o ar. Foi utilizado como convenção que o trabalho é positivo quando é
    aplicado PELO sistema SOBRE o ambiente."""

def acha_w_real(rel_compress , efic_isoentrop): # Dado e kJ/kmol

    T1 = 25 + 273.15 # temperatura padrão de 25°C em Kelvin
    T2_ideal = acha_T2_ideal(rel_compress) # temperatura em 2 ideal

    w_ideal = del_h(T1,T2_ideal) #trabalho ideal para temperatura de 2

    w_real = w_ideal/efic_isoentrop #trabalho real do compressor

    return w_real #kJ/kmol


'---------------------------------------------------------------------------'

"""Determinação da temperatura REAL de saída do ar T2, considerando o trabalho
   real corrigido do trabalho isoentrópico"""

def acha_T2_real(w_real): # dado em Kelvin


    def func(x,args = None): #Função que reprensenta o sistema de equações
                             #em que h2 - h1 - w_real = 0 [processo real]
                             
        T1 = 25 + 273.15 #temperatura padrão de 25°C em Kelvin
        T2 = x           #nossa incógnita
 

        return del_h(T1,T2) - w_real
    
    #resolução do sistema de eq's
    T2_real = robustNR_args.robustNewton(func,300.) 

    return T2_real[0] # dado em Kelvin

'---------------------------------------------------------------------------'


"""Finalmente, a função que resolve o item a., com relação de compressão
   e a eficiência isoentrópica do compressor e como saída teremos a tem-
   peratura e trabalho por kmol de ar na situação ideal e na situação real"""

def resolve_item_a(rel_compress , efic_isoentrop): #dados de entrada

     'Cálculo do trabalho real no compressor e a temperatura de saída do ar'    
     w_real = acha_w_real(rel_compress , efic_isoentrop) # kJ/kmol
     T2_real = acha_T2_real(w_real)                      # K
     
     return { 'temperatura_T2':T2_real , 'trabalho_w': w_real} #dados de saída


"--------------- FUNÇÔES QUE CALCULAM OS DADOS PARA O ITEM B. --------------"

"""Determinação de x vezes de ar teórico (1/phi) para uma dada temperatura
   adiabática de chama na câmara de combustão"""

def acha_X(T2_ar,T_adiab,rel_compress): 


    def func(x,args = None): #Função que reprensenta o sistema de equações
                             #envolvendo balanço de massas e energia na câmara
                             #de combustão

        X , a_s = x #nossas incógnitas são o x vezes de ar teorico e a
                    #quantidade estequiométrica de ar

        'Constantes úteis'
        P1 = 100. #kPa
        P2 = rel_compress * P1  #kPa                   
        T0 = 25 + 273.15 #temperatura 25°C em Kelvin

        'Massas Moleculares'
        M_ar  = 28.97 #kg/kmol
        M_NH3 = 17.031 #kg/kmol
        M_mist = equi.calcula_prods(0.,3.,0.,1.,1./X,T_adiab,P2, jacob = 1)[6] #kg/kmol
        


        def h_mist(X,T,P): #uma função mais conveniente da entalpia dos produtos
                           #da combustão

            phi = 1./X # relação de equivalência e x
            M_mist = equi.calcula_prods(0.,3.,0.,1.,phi,T,P, jacob = 1)[6] #kg/kmol
            h_mist = equi.calcula_prods(0.,3.,0.,1.,phi,T,P, jacob = 1)[3] #kJ/kg 
            h_mist = h_mist * M_mist #kJ/kmol

            return h_mist

        """A seguir as quantidades de kmoles entrando e saindo do volume de
           controle em função de x e a_s"""
        n_ar = 4.76 * X * a_s #kmol
        n_NH3 = 2. #kmol
        n_mist = 2 * equi.calcula_prods(0.,3.,0.,1.,1./X,T_adiab,P2, jacob = 1)[1] #kmol

        'No caso -80800kJ/kmol é a entalpia de formação de NH3 líquido'
        return array (( n_ar * del_h(T0,T2_ar) + 2 * (-80800) - n_mist * h_mist(X,T_adiab,P2),
                        M_ar * n_ar + M_NH3 * n_NH3 - M_mist * n_mist,
                        )  )#duas equações a duas incógnitas (balanços)
    
    'resolução do sistema de equações'
    X = robustNR_args.robustNewton(func,(1.2,1.2))

    return X[0][0] 


'---------------------------------- PRINCIPAL ---------------------------------'

if __name__ == '__main__':

    print """\n
   Programa desenvolvido para resolver o exercício 12.81 do livro Fundamentos
   da Termodinâmica, 4ªed., de Van Wylen et alli. Abaixo o enunciado do pro-
   blema:

   12.81 Escreva um programa de computador para resolver uma generalização do
   prob. 12.37. Utilize a relação de compressão, a eficiência isoentrópica do
   compressor e a temperatura adiabática de chama como variáveis de entrada
   do programa.

   12.37 Um estudo está sendo realizado para avaliar se a amônia líquida é um
   combustível adequado para uma turbina a gás. Considere os processos de com-
   pressão e combustão deste equipamento.

   a. Ar entra no compressor a 100kPa e 25°C. Este é comprimido até 1600kPa e
   a eficiência isoentrópica é 87%. Determine a temperatura de descarga do ar
   e o trabalho consumido por kmol de ar.

   b. Dois kmoles de amônia líquida a 25°C e x vezes do ar teórico, proveni-
   entes do compressor, entram na câmara de combustão. Qual o valor de x se a
   temperatura adiabática de chama for fixada em 1600 K."""
 
    print """\n
    Entre com o valor da relação de compressão(pressão de saída
    pela de entrada (P2/P1), p.e., 16.1). Lembrando que a
    pressão de entrada é de 100kPa:\n"""

    rel_compress = float(raw_input('Relação de compressão:'))

    print """\n
    Entre com o valor eficiência isoentrópica do compressor (p.e., ef=0.87):\n"""

    efic_isoentrop = float(raw_input('Eficiência isoentrópica:'))

    print """\n
    Entre com a temperatura adiabática de chama na câmara de combustão
    em kelvins (p.e., 1600). Este programa tem uma amplitude de tempera-
    turas aceitas de 300K até 3000K :\n"""

    T_adiab = float(raw_input('Temperatura adiabática:'))

temp = resolve_item_a(rel_compress , efic_isoentrop)['temperatura_T2']
trab = resolve_item_a(rel_compress , efic_isoentrop)['trabalho_w']
x    = acha_X( temp , T_adiab , rel_compress)

print """\nA partir de uma relação de compressão de %.2f , eficiência
isoentrópica de %.3f e temperatura adiabática de %.1fK,
obteremos como resultados:""" %(rel_compress,efic_isoentrop,T_adiab)

print "\nTemperatura de saída do ar no compressor(K):" , temp
print "\nTrabalho real realizado pelo compressor (kJ/kmol):" , trab
print "\nValor de x vezes de ar teórico:" , x

print "\nObrigado pela atenção!"





        
        
