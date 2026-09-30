"""Independent finite-dimensional Chapter 6 reference (no finite-memory window).

Equations follow the September 29 math handoff. This is a research/reference
implementation, separate from the general noisestate solver. Selects the branch
with positive trading intensity, mean-reverting inventory and stabilizing MM
feedback; every returned root is checked against the original equations.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import root
from scipy.linalg import expm

@dataclass
class Equilibrium:
    market: str
    params: dict
    root: np.ndarray
    beta: float
    delta: float
    pq: float
    residual: float
    mm_gain: np.ndarray
    mm_drift: np.ndarray
    blip_initial: np.ndarray

    @property
    def costs(self):
        e,g,s=self.params['eps'],self.params['gamma'],self.params['sigma_Z']
        gross=s-self.pq*s*s/2
        trading=e*(self.beta*s+self.delta*s*s/2)
        inventory=g*s*s/(2*self.delta) if g else 0.
        return {'market_maker':gross+inventory,'trader':trading-gross}

    def kernels(self, ages):
        s=self.params['sigma_Z']; b,d,p=self.beta,self.delta,self.pq; l=1/s
        F=np.array([[0,0,0,0],[1,-1,0,0],[0,l*b,-l*b,0],[0,-b,b,-d]],float)
        B=np.array([[1,0,0],[0,0,1],[0,l*s,0],[0,-s,0]],float)
        C=np.array([[1,0,0,0],[0,0,0,1],[0,0,1,p],[0,b,-b,d]],float)
        out=np.array([C@expm(F*t)@B for t in ages])
        return {k:out[:,i,:] for i,k in enumerate(('V','Q','P','D'))}

    def path_reference(self, h=.1):
        # Integrate the covariance directly, independently of the JS block exponential.
        from scipy.integrate import quad_vec
        s = self.params['sigma_Z']; b,d,p = self.beta,self.delta,self.pq
        F = np.array([[-1,0,0,0],[1,-b/s,0,0],[0,-b,-d,0],[0,0,0,0]],float)
        B = np.array([[1,0,-1],[0,-1,1],[0,-s,0],[1,0,0]],float)
        G = B@B.T
        initial = np.diag([1,s/b,s*s/(2*d) if self.params['gamma'] else 0,0])
        C = np.array([[0,0,0,1],[0,0,1,0],[-1,-1,p,1],[0,b,d,0]],float)
        def covariance(t):
            noise, _ = quad_vec(lambda u: expm(F*u)@G@expm(F.T*u), 0, t, epsabs=1e-12, epsrel=1e-12)
            A = expm(F*t)
            return A@initial@A.T+noise, noise
        moments = {str(t): np.diag(C@covariance(t)[0]@C.T).tolist() for t in (0,1,40)}
        return dict(transition=expm(F*h).tolist(), noise_covariance=covariance(h)[1].tolist(),
                    initial_covariance=initial.tolist(), variances=moments)

    def deviation(self, ages):
        if self.market=='transparent':
            q=self.blip_initial[0]*np.exp(-self.delta*np.asarray(ages))
            return {'Q':q,'P':self.pq*q,'D':self.delta*q}
        states=np.array([expm(self.mm_drift*t)@self.blip_initial for t in ages])
        return {'Q':states[:,0], 'P':self.pq*states[:,0]-states@self.mm_gain,
                'D':-states@self.mm_drift[0,:]}


def solve(market='transparent', eps=.2, gamma=.1, rho=.5, sigma_Z=1., seed=None):
    if market not in ('transparent','opaque'): raise ValueError(market)
    if min(eps,rho,sigma_Z)<=0 or gamma<0: raise ValueError('positive eps, rho, sigma_Z and nonnegative gamma required')
    lam=1/sigma_Z
    def trader(z,pq):
        a,b,c=z[:3]; n1=1-lam*a-b; n2=-pq-lam*b-c
        beta=n1/(2*eps);delta=n2/(2*eps)
        return [rho*a/2-n1*n1/(4*eps),rho*b-n1*n2/(2*eps)-lam*delta*a,
                rho*c/2-n2*n2/(4*eps)-lam*delta*b],beta,delta
    def mm(z,pq,beta,delta,g):
        u11,u12,u22=z[3:6]
        cn=(beta*lam+delta)/(lam-pq);cx=beta*pq+delta
        kx=lam*beta*pq/(lam-pq);kn=lam*lam*beta/(lam-pq)**2
        A=np.array([[-delta,-cx],[0,kx]]);B=np.array([-cn,kn])
        R=np.array([[-pq*delta+g,-pq*cx/2],[-pq*cx/2,0]])
        N=np.array([(delta-cn*pq)/2,cx/2]);U=np.array([[u11,u12],[u12,u22]])
        v=N+U@B;K=-v/cn
        E=rho*U-(R+A.T@U+U@A-np.outer(v,v)/cn)
        return [E[0,0],E[0,1],E[1,1]],K,A+np.outer(B,K),np.array([cn,-kn])
    def terms(z,g):
        if market=='transparent':
            a,b,c,u=z;gq=-(lam*b+c)/(2*eps);pq=eps*gq-u/2
            eq,beta,delta=trader(z,pq)
            eq.append(rho*u-2*g+eps*gq*gq+gq*u+u*u/(4*eps))
            return np.array(eq),beta,delta,pq,np.zeros(1),np.array([[-delta]]),np.array([1/(2*eps)])
        pq=z[6];eq,beta,delta=trader(z,pq);er,K,A,initial=mm(z,pq,beta,delta,g)
        return np.array(eq+er+[K[0]]),beta,delta,pq,K,A,initial
    a0=1/(lam+eps*rho+np.sqrt(eps*rho*(2*lam+eps*rho)))
    z=np.array([a0,0,0,0] if market=='transparent' else [a0,0,0,0,0,0,0],float)
    stages=np.linspace(0,gamma,max(2,int(np.ceil(gamma/.01))+1))[1:] if gamma else [0]
    if seed is not None:
        z=np.asarray(seed,float);stages=[gamma]
    for g in stages:
        sol=root(lambda q:terms(q,g)[0],z,tol=1e-11)
        vals=terms(sol.x,g);err=float(np.max(np.abs(vals[0])))
        if not np.isfinite(err) or err>1e-9 or vals[1]<=0 or (g>0 and (vals[2]<=0 or vals[3]>=0 or np.max(np.linalg.eigvals(vals[5]).real)>1e-9)):
            raise RuntimeError(f'{market} invalid equilibrium at gamma={g}: residual={err}, beta={vals[1]}, delta={vals[2]}, pq={vals[3]}, {sol.message}')
        z=sol.x
    _,beta,delta,pq,K,A,initial=vals
    return Equilibrium(market,dict(eps=eps,gamma=gamma,rho=rho,sigma_Z=sigma_Z),z,beta,delta,pq,err,K,A,initial)

if __name__=='__main__':
    import json
    for market in ('transparent','opaque'):
        for pt in ({},{'gamma':0.},{'gamma':.01},{'gamma':.2},{'eps':.1},{'eps':1.},{'rho':.2},{'rho':2.},{'sigma_Z':.3},{'sigma_Z':3.}):
            eq=solve(market,**pt)
            print(json.dumps({'market':market,'point':pt,'beta':eq.beta,'delta':eq.delta,'pq':eq.pq,'residual':eq.residual,'costs':eq.costs,'root':eq.root.tolist()}),flush=True)
