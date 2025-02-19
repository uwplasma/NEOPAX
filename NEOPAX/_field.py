import h5py as h5 
import numpy as np
import jax.numpy as jnp
from jax import jit
from netCDF4 import Dataset
import interpax 
from _parameters import vmec,booz


#This should go to equilibrium reader
vfile = Dataset(vmec, mode="r")
bfile = Dataset(booz, mode="r")

ns = vfile.variables["ns"][:].filled()
s_full = jnp.linspace(0,1,ns)  #This is s_full
s_half_list = [(i-0.5)/(ns-1) for i in range(0,ns)] #This is s_half
s_half =jnp.array(s_half_list)

rho_half=jnp.sqrt(s_half)
rho_full=jnp.sqrt(s_full)

Vprime = vfile.variables["vp"][:].filled()
Aminor_p = vfile.variables["Aminor_p"][:].filled()   
volume_p = vfile.variables["volume_p"][:].filled()
vp = vfile.variables["vp"][:].filled()  
iotaf = vfile.variables["iotaf"][:].filled()                              
phi = vfile.variables["phi"][:].filled()  
Psia=jnp.abs(phi[-1])

bmnc_b=bfile.variables["bmnc_b"][:].filled() 
rmnc_b=bfile.variables["rmnc_b"][:].filled()
gmnc_b=bfile.variables['gmn_b'][:].filled()
xm_b=bfile.variables['ixm_b'][:].filled()
xn_b=bfile.variables['ixn_b'][:].filled()
bvco=bfile.variables['bvco_b'][:].filled()
buco=bfile.variables['buco_b'][:].filled()

R0_b=rmnc_b[-1,0]
a_b=np.sqrt(volume_p/(2*np.pi**2*R0_b))


for l in range(len(xm_b)):
    if(xm_b[l]==0 and xn_b[l]==0):
        B00=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)
        R00=interpax.Interpolator1D(rho_full[1:],rmnc_b[:,l],extrap=True)
        sqrtg00=interpax.Interpolator1D(rho_half[1:],gmnc_b[:,l],extrap=True)        
    if(xm_b[l]==1 and xn_b[l]==0):
        B10=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)

dVdr=interpax.Interpolator1D(rho_half[1:],vp[1:],extrap=True)
iota=interpax.Interpolator1D(rho_full[:],iotaf[:],extrap=True)
G=interpax.Interpolator1D(rho_half[1:],bvco[1:],extrap=True)
I=interpax.Interpolator1D(rho_half[1:],buco[1:],extrap=True)


