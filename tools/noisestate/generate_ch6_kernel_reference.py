"""Independent high-precision propagation fixtures for difficult rate spacings.

These test the state propagator, not equilibrium selection. Synthetic state
coefficients isolate rate coincidences that random equilibrium draws can miss.
"""
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from scipy.linalg import expm

def product(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))),Decimal(0))
             for j in range(len(b[0]))] for i in range(len(a))]

def decimal_matrix(a):
    return [[Decimal(float(x)) for x in row] for row in a]

def precise_expm(a,t):
    """80-digit Taylor scaling/squaring in the original state coordinates.

    This does not use the browser's scalar convolution formulas. High precision
    avoids the near-coincident-rate error observed in the SciPy reference.
    """
    n=len(a);norm=float(np.abs(a*t).sum(axis=1).max())
    squarings=max(0,math.ceil(math.log2(max(1.,norm*2))))
    scale=Decimal(float(t))/Decimal(2)**squarings
    b=[[x*scale for x in row] for row in decimal_matrix(a)]
    out=[[Decimal(int(i==j)) for j in range(n)] for i in range(n)]
    term=[row[:] for row in out]
    for order in range(1,100):
        term=[[x/Decimal(order) for x in row] for row in product(term,b)]
        out=[[out[i][j]+term[i][j] for j in range(n)] for i in range(n)]
        if max(abs(x) for row in term for x in row)<Decimal('1e-70'):break
    else:raise RuntimeError('high-precision exponential did not converge')
    for _ in range(squarings):out=product(out,out)
    return out

def main():
    cases=[];worst_scipy=0.
    rates=[(1.,1.),(1.+1e-12,1.-1e-12),(1.+1e-4,1.-1e-4),
           (.03,.031),(.5,.5),(20.,1e-6),(1e-8,1e-8),(0.,0.)]
    for i,(k,d) in enumerate(rates):
        s=(.2,1.,3.)[i%3];b=k*s;p=-.3
        ages=[0.,1e-10,1e-7,1e-4,.01,.1,.5,1.,2.,10.,40.,160.,1000.]
        A=np.array([[-d,.3],[0.,-d-1e-12]])
        initial=np.array([1.,-.5]);gain=np.array([0.,.17])
        if i==2:A=np.array([[-.1,2.],[-2.,-.1]])
        if i==3:A=np.array([[-1.,1e4],[0.,-1.]])
        if i==4:
            A=np.array([[-1e-8,.2],[0.,-100.]])
            ages += [1e6,1e8]
        F=np.array([[0,0,0,0],[1,-1,0,0],[0,k,-k,0],[0,-b,b,-d]],float)
        B=np.array([[1,0,0],[0,0,1],[0,1,0],[0,-s,0]],float)
        C=np.array([[1,0,0,0],[0,0,0,1],[0,0,1,p],[0,b,-b,d]],float)
        # Compute in the original state ordering rather than the eliminated cascade.
        with localcontext() as ctx:
            ctx.prec=80
            K=np.array([product(decimal_matrix(C),product(precise_expm(F,t),decimal_matrix(B))) for t in ages],float)
            state=np.array([product(precise_expm(A,t),decimal_matrix(initial[:,None])) for t in ages],float)[:,:,0]
        scipy_K=np.array([C@expm(F*t)@B for t in ages])
        worst_scipy=max(worst_scipy,float(abs(K-scipy_K).max()))
        deviation=dict(Q=state[:,0].tolist(),P=(p*state[:,0]-state@gain).tolist(),
                       D=(-state@A[0]).tolist())
        cases.append(dict(name=f'rates-{i}',ages=ages,
                          equilibrium=dict(beta=b,delta=d,pq=p,params=dict(sigma_Z=s),
                                           drift=A.tolist(),initial=initial.tolist(),gain=gain.tolist()),
                          kernels={name:K[:,j,:].tolist() for j,name in enumerate(('V','Q','P','D'))},
                          deviation=deviation))
    Path(__file__).with_name('ch6-kernel-reference.json').write_text(json.dumps(
        dict(reference='80-digit matrix exponential',scipy_max_difference=worst_scipy,cases=cases),indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),scipy_max_difference=worst_scipy)))

if __name__=='__main__':main()
