from ch6_reference import solve
import json,itertools,pathlib
points=[dict(eps=e,gamma=g,rho=r,sigma_Z=s) for e,g,r,s in itertools.product((.1,1.),(0.,.01,.2),(.2,2.),(.3,3.))]
points+=[dict(eps=.2,gamma=.1,rho=.5,sigma_Z=1.)]
ages=[0,.1,1,10,40,1000]
fixtures=[]
for market in ('transparent','opaque'):
 for params in points:
  eq=solve(market,**params)
  fixtures.append({'market':market,'params':params,'root':eq.root.tolist(),'costs':eq.costs,
                   'kernels':{k:v.tolist() for k,v in eq.kernels(ages).items()},
                   'deviation':{k:v.tolist() for k,v in eq.deviation(ages).items()}, 'paths':eq.path_reference()})
pathlib.Path(__file__).with_name('ch6-reference.json').write_text(json.dumps({'source':'Independent SciPy root and matrix-exponential implementation; 2026-09-30','ages':ages,'cases':fixtures},separators=(',',':'))+'\n')
