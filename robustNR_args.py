# -*- coding: cp1252 -*-
"""
Uma implementação direta do método de Newton-Raphson tornada robusta pelo
controle do avanço de cada componente do vetor dx, para manter cada x
sempre positivo. fun é a função vetorial ou escalar de x (cada x >=o),
x0 são as condições iniciais e jacob é a função jacobiana (ou a derivada)
obtida algébricamente ou numericamente por diferenças finitas,
se jacob is None. Esta versão permite argumentos-extra na função e está
totalmente vetorizada. 
"""
#                                                           #
#   Por E R Woiski UNESP Ilha Solteira - SP - 2007-09-17    #
#                                                           #

from numpy import array,any,dot,size,asarray,zeros,identity,ones,copy
from numpy.linalg import solve
import time

def robustNewton(fun,x0,jacob = None, nitermax = 200, xtol=1.e-8, args=None):
    
    error = 2*xtol
    ite = 0
    if args:
        args = args
        
    if size(x0) != 1: #  VETORIAL!...
        x = asarray(x0)
        linhas = colunas = size(x0)        
       
        while error > xtol and ite <= nitermax:
            F = fun(x, args)              
            if jacob is None:
                ''' se o jacobiano não for fornecido,
                calcule um por diferenças finitas...'''
                ix = ones((linhas,colunas))*x
                ixtol = identity((colunas))*xtol
                J = (array(map(lambda x:fun(x,args),ix + ixtol)) -
                  array(map(lambda x:fun(x,args),ix)) ).transpose()/xtol      
           
            else: # ou então jacobiano fornecido...
                J = jacob(x,args)
                        
            # seja qual for a forma de determinação do jacobiano...    
            dx = solve(J,-F)
            
            # se algum componente de xi + dxi for negativo, reduza dxi pela metade...
            while dx[(x+dx)<0].size != 0:
                dx[(x+dx)<0] *= .5  
                        
            ite += 1
            x += dx # avança x...
            F = fun(x, args)
            
            # O erro é o maior valor dentre a norma quadrática e o maior resíduo...
            error = max(dot(F,F),abs(F).max())

    else: #ESCALAR!...
        x = x0       
        while error > xtol and ite <= nitermax:
            F = fun(x, args)                                                                   
            if jacob is None:
                J = (fun(x*(1.+xtol),args) - fun(x,args))/(x*xtol)
                ''' se o jacobiano não for fornecido,
                    calcule um por diferenças finitas...'''
                                                   
            else: # ou então jacob fornecido...
                J = jacob(x,args)
                        
            # seja qual for a forma de determinação do jacobiano...    
            dx = -F/J

            # se x + dx for negativo, reduza dx pela metade...                                                         
            while -dx > x: dx *= .5

            ite += 1
            x += dx # avança x...
            F = fun(x, args)
                
            error = abs(F)

    if ite >= nitermax:
        print '%s: não convergiu com %s iterações!' %(fun,nitermax)
        
    return x,ite,F

if __name__ == '__main__':  # exemplos de utilização...
    from math import sin,cos,log

    print 'Doc string do módulo robustNR:'
    import robustNR_args as NR
    print NR.__doc__

    args = (-1.,0.5)
##    args = (0.,1)
    
    def fun(x,args):
        """
        Define a função vetorial multivariável que vai representar o
        sistema de equações. Note que x é o vetor das incógnitas
        x1, x2 e x3 (que devem ser todas maior (menor) ou igual a zero...)
	"""
        x1, x2, x3 = x
        arg1,arg2 = args

        return array((
                    x1**2 - sin(2*x2)  - .3*log(x3) - 0.42721880871 + arg1,
                    x2**3 + cos(2.*x1) + x3**(-1/2.) - 4.08060171632/arg2,
                    0.5*x1**(2./3) + x2**(2./7) - cos(x3) - 2.7090061508,
                      ) )
    
    def jacob(x,args):
        x1, x2, x3 = x
        return array((
                [2.*x1, - 2.*cos(2*x2), -.3/x3 ],
                [- 2.*sin(2.*x1), 3.*x2**2, -1./2*x3**(-3/2)],
                [1/3.*x1**(-1./3), 2./7*x2**(5./7), +sin(x3)]
                    ))

    x,ite,F = robustNewton(fun,(1.,1.,1.),args=args,jacob = jacob)
    xs,ites,Fs = robustNewton(fun,(1.,1.,1.),jacob = None, args = args)

    print 'EXEMPLOS SIMPLES:'
    print '\n1- Sistema não-linear resolvido por Newton-Raphson:'
    print '''
            Para arg1,arg2 = (-1.,0.5):
            
            x1**2 - sin(2*x2)  - .3*log(x3) - 0.42721880871 + arg1 = 0.
            x2**3 + cos(2.*x1) + x3**(-1/2.) - 4.08060171632/arg2 = 0.
            0.5*x1**(2./3) + x2**(2./7) - cos(x3) - 2.7090061508 = 0.           
          '''
    print 'raízes:\ncom jacobiano:',x,' em ',ite,
    print 'iterações\ne sem jacobiano:',xs,' em ',ite,'iterações'
    print 'resíduos:\ncom jacobiano:',F,'\ne sem jacobiano:',Fs
    print 50*'*'
    
    def fun_linear(x,arg):
        x1,x2 = x
        return array((
                x1 + 2.*x2 -5.,
                3.*x1 + 4.*x2 - 11.
                    ))

    def jacob_linear(x,arg):
        x1,x2 = x
        return array((
                [1.,2.],
                [3.,4.]
                    ))
    
    xlin,itlin,Flin = robustNewton(fun_linear,(1.,1.),jacob = jacob_linear)
    xlins,itlins,Flins = robustNewton(fun_linear,(1.,1.))
    
    print '\n2- Sistema linear resolvido por Newton-Raphson:'
    print '''
            x1 + 2.*x2 -5. = 0.
            3.*x1 + 4.*x2 - 11. = 0.        
          '''
    print 'raízes:\ncom jacobiano:',xlin,' em ',itlin,
    print 'iterações\ne sem jacobiano:',xlins,' em ',itlins,'iterações'
    print 'resíduos:\ncom jacobiano:',Flin,'\ne sem jacobiano:',Flins

    a = array((
              [1.,2.],
              [3.,4.]
              ))

    b = array((5.,11.))
    xlin2 = solve(a,b)
    print '\nsistema linear resolvido por solução direta:'
    print 'raízes: ', xlin2
    print '*'*50

    def calcula_taxa(A,R,n):
        """
Conhecidos o Valor Atual (A), a prestação (R) e o número de prestações (n),
a função calcula a taxa real de juros (i), retornando i e o resíduo F.
        """
        def FVAs(i,n):
            fac = (1 + i)**n
            return (fac - 1.)/(i*fac) - A/R

        ires,ite,F = robustNewton(FVAs,1., args=n)

        return ires,ite,F

    A,R,n = 3419.,721.21,6

    i,ite,F = calcula_taxa(A,R,n)

    print '\n3- Exemplo de função escalar:\nCálculo de Taxas de Juros Compostos:'
    print 'Doc string da função calcula_taxa:'
    print calcula_taxa.__doc__,
    print '''
            Observe que n entra como argumento-extra na função...
            
            FVAs(i,n):
            fac = (1 + i)**n
            return (fac - 1.)/(i*fac) - A/R           
          '''
    print """Valor Atual:\tR$%.2f
Prestação:\tR$%.2f
número de prestações:\t%s\n""" %(A,R,n)
    
    print 'Taxa de juros:\t%f'%(i*100) + ' %',' em ',ite,'iterações\n'
    print 'resíduo: %s' %F
    print '*'*50
