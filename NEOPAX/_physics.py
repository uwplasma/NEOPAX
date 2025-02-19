import jax.numpy as jnp
import jax
from jax import jit
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
import interpax
from scipy.constants import proton_mass,elementary_charge
from _grid import rho_grid,rho_grid_half, r_grid
from _field import B00, B10, a_b, R00, iota, dVdr ,G,I
from _species import coulomb_logarithm


epsilon_t=rho_grid*a_b/R00(rho_grid)
B0=B00(rho_grid_half)
B_10=B10(rho_grid_half)/B0
enlogation=jnp.square(epsilon_t/B_10)
psi_fac=1.+1./(enlogation*jnp.square(iota(rho_grid)))
psi_fac=psi_fac.at[0].set(1.)
#Important geometrical quantities interpolated from equilibrium
 
@jit
def get_plasma_permitivity(species,grid_x):    
    mass_density=species.mass[0]*species.density[0,:]+species.mass[1]*species.density[1,:]+species.mass[2]*species.density[2,:]
    epsilon_r=mass_density*psi_fac/jnp.square(B0)
    plasma_permitivity=interpax.Interpolator1D(r_grid,epsilon_r,extrap=True)
    #psi_fac=psi_fac.at[0].set(0)
    return plasma_permitivity(grid_x)


#These should go to the physics_models.py 
#Get FusionPower Fraction to Electrons, using same model as NTSS - update in the future
def FusionPowerFractionElectrons(species,r_index): 
    Te=species.temperature[0,r_index]*1.e-3
    y2 = 88./Te
    y = jnp.sqrt(y2)
    part = 2.*(jnp.log((1-y+y2)/(1+2*y+y2))/6 + 0.57735026*jnp.atan(0.57735026*(2*y-1))+0.30229987 ) / y2
    return 1.-part

#Get D-T reaction rate, using same model as NTSS - update in the future
def get_DT_Reaction(species, r_index):
    nD=species.density[1,r_index]*1.e-20 
    TT=species.temperature[2,r_index]*1.e-3
    nT=species.density[2,r_index]*1.e-20
    t = jnp.pow(TT,-1./3.)
    wrk = (TT+1.0134)/(1+6.386e-3*jnp.square(TT+1.0134))+1.877*jnp.exp(-0.16176*jnp.sqrt(TT)*TT)
    DTreactionRate= 8.972e-19*t*t*jnp.exp(-19.94*t)*wrk  ##D-T reaction rate in m^3/sec
    HeSource=1e20*DTreactionRate*nD*nT  #in 10^20/m^3/s
    AlphaPower=3.52e3*HeSource  #in keV*10^20/m^3/s
    return DTreactionRate,HeSource,AlphaPower


#Get Power exchange
def Power_Exchange(species,species_a,species_b,r_index):
    nA=species.density[species_a,r_index]*1.e-20
    nB=species.density[species_b,r_index]*1.e-20
    TA=species.temperature[species_a,r_index]*1.e-3
    TB=species.temperature[species_b,r_index]*1.e-3
    mA=species.mass[species_a]/proton_mass
    mB=species.mass[species_b]/proton_mass
    qA=species.charge[species_a]
    qB=species.charge[species_b]
    lnL = coulomb_logarithm(species_a,species_b,species,r_index)
    Pab=663.*jnp.sqrt(mA*mB)*jnp.square(qA*qB/(elementary_charge*elementary_charge))*nA*nB*lnL*(TB-TA)/jnp.pow(mA*TB+mB*TA,1.5)  #Power exchange in keV*10^20/m^3/s
    return Pab



def P_rad(species,r_index):
    Te=species.temperature[0,r_index]*1.e-3
    ne=species.density[0,r_index]*1.e-20
    nD=species.density[1,r_index]*1.e-20
    nT=species.density[2,r_index]*1.e-20
    ZD=species.charge[1,r_index]/elementary_charge
    ZT=species.charge[2,r_index]/elementary_charge
    Zeff=(ZD*ZD*nD + ZT*ZT*nT)/ne       #(ZD*ZD*ND + ZT*ZT*NT + ZHe*ZHe*NHe)/Ne
    PBrems=3.16e-1*Zeff*ne*ne*jnp.sqrt(Te)       # in keV*10^20/m^3/s, from ASTRA 
    return PBrems,Zeff

