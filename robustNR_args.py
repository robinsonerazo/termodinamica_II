# -*- coding: cp1252 -*-
"""
Uma implementa��o direta do m�todo de Newton-Raphson tornada robusta pelo
controle do avan�o de cada componente do vetor dx, para manter cada x
sempre positivo. fun � a fun��o vetorial ou escalar de x (cada x >=o),
x0 s�o as condi��es iniciais e jacob � a fun��o jacobiana (ou a derivada)
obtida alg�bricamente ou numericamente por diferen�as finitas,
se jacob is None. Esta vers�o permite argumentos-extra na fun��o e est�
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
                ''' se o jacobiano n�o for fornecido,
                calcule um por diferen�as finitas...'''
                ix = ones((linhas,colunas))*x
                ixtol = identity((colunas))*xtol
                J = (array(map(lambda x:fun(x,args),ix + ixtol)) -
                  array(map(lambda x:fun(x,args),ix)) ).transpose()/xtol      
           
            else: # ou ent�o jacobiano fornecido...
                J = jacob(x,args)
                        
            # seja qual for a forma de determina��o do jacobiano...    
            dx = solve(J,-F)
            
            # se algum componente de xi + dxi for negativo, reduza dxi pela metade...
            while dx[(x+dx)<0].size != 0:
                dx[(x+dx)<0] *= .5  
                        
            ite += 1
            x += dx # avan�a x...
            F = fun(x, args)
            
            # O erro � o maior valor dentre a norma quadr�tica e o maior res�duo...
            error = max(dot(F,F),abs(F).max())

    else: #ESCALAR!...
        x = x0       
        while error > xtol and ite <= nitermax:
            F = fun(x, args)                                                                   
            if jacob is None:
                J = (fun(x*(1.+xtol),args) - fun(x,args))/(x*xtol)
                ''' se o jacobiano n�o for fornecido,
                    calcule um por diferen�as finitas...'''
                                                   
            else: # ou ent�o jacob fornecido...
                J = jacob(x,args)
                        
            # seja qual for a forma de determina��o do jacobiano...    
            dx = -F/J

            # se x + dx for negativo, reduza dx pela metade...                                                         
            while -dx > x: dx *= .5

            ite += 1
            x += dx # avan�a x...
            F = fun(x, args)
                
            error = abs(F)

    if ite >= nitermax:
        print '%s: n�o convergiu com %s itera��es!' %(fun,nitermax)
        
    return x,ite,F

if __name__ == '__main__':  # exemplos de utiliza��o...
    from math import sin,cos,log

    print 'Doc string do m�dulo robustNR:'
    import robustNR_args as NR
    print NR.__doc__

    args = (-1.,0.5)
##    args = (0.,1)
    
    def fun(x,args):
        """
        Define a fun��o vetorial multivari�vel que vai representar o
        sistema de equa��es. Note que x � o vetor das inc�gnitas
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
    print '\n1- Sistema n�o-linear resolvido por Newton-Raphson:'
    print '''
            Para arg1,arg2 = (-1.,0.5):
            
            x1**2 - sin(2*x2)  - .3*log(x3) - 0.42721880871 + arg1 = 0.
            x2**3 + cos(2.*x1) + x3**(-1/2.) - 4.08060171632/arg2 = 0.
            0.5*x1**(2./3) + x2**(2./7) - cos(x3) - 2.7090061508 = 0.           
          '''
    print 'ra�zes:\ncom jacobiano:',x,' em ',ite,
    print 'itera��es\ne sem jacobiano:',xs,' em ',ite,'itera��es'
    print 'res�duos:\ncom jacobiano:',F,'\ne sem jacobiano:',Fs
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
    print 'ra�zes:\ncom jacobiano:',xlin,' em ',itlin,
    print 'itera��es\ne sem jacobiano:',xlins,' em ',itlins,'itera��es'
    print 'res�duos:\ncom jacobiano:',Flin,'\ne sem jacobiano:',Flins

    a = array((
              [1.,2.],
              [3.,4.]
              ))

    b = array((5.,11.))
    xlin2 = solve(a,b)
    print '\nsistema linear resolvido por solu��o direta:'
    print 'ra�zes: ', xlin2
    print '*'*50

    def calcula_taxa(A,R,n):
        """
Conhecidos o Valor Atual (A), a presta��o (R) e o n�mero de presta��es (n),
a fun��o calcula a taxa real de juros (i), retornando i e o res�duo F.
        """
        def FVAs(i,n):
            fac = (1 + i)**n
            return (fac - 1.)/(i*fac) - A/R

        ires,ite,F = robustNewton(FVAs,1., args=n)

        return ires,ite,F

    A,R,n = 3419.,721.21,6

    i,ite,F = calcula_taxa(A,R,n)

    print '\n3- Exemplo de fun��o escalar:\nC�lculo de Taxas de Juros Compostos:'
    print 'Doc string da fun��o calcula_taxa:'
    print calcula_taxa.__doc__,
    print '''
            Observe que n entra como argumento-extra na fun��o...
            
            FVAs(i,n):
            fac = (1 + i)**n
            return (fac - 1.)/(i*fac) - A/R           
          '''
    print """Valor Atual:\tR$%.2f
Presta��o:\tR$%.2f
n�mero de presta��es:\t%s\n""" %(A,R,n)
    
    print 'Taxa de juros:\t%f'%(i*100) + ' %',' em ',ite,'itera��es\n'
    print 'res�duo: %s' %F
    print '*'*50
