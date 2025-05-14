from functools import partial
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx
import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import jax.numpy as jnp
from ._constants import Boltzmann, elementary_charge, epsilon_0, hbar, proton_mass
import interpax
#from _io import grid,field



###This module uses some functions adapted from JAX-MONKES by R. Colin, but class structure was revamped to fit the overall package better#####

JOULE_PER_EV = 11606 * Boltzmann
EV_PER_JOULE = 1 / JOULE_PER_EV

@jit
def get_v_thermal(mass,temperature):
    return jnp.sqrt(2*temperature * JOULE_PER_EV/mass)

@jit
def get_gradient_temperature(T,r_grid,r_grid_half,dr,T_edge):
    T_next = jnp.roll(T, shift=-1)
    #T_prev = jnp.roll(T, shift=1)    
    #grad_y = (y_next -  y_prev) / (2.*y.delta_x)
    grad_T = (T_next -  T) / (dr)
    dTdr_full=interpax.Interpolator1D(r_grid_half[:-1],grad_T[:-1])(r_grid)
    #SET gradients to zero at radial coordinate r=0 
    dTdr_full = dTdr_full.at[0].set(0.)
    dTdr_full = dTdr_full.at[-1].set((4.*T.at[-2].get()-T.at[-3].get()-3*T_edge)/(-2*dr))   
    return dTdr_full

@jit
def get_gradient_density(n,r_grid,r_grid_half,dr,n_edge):
    n_next = jnp.roll(n, shift=-1)
    #n_prev = jnp.roll(n, shift=1)    
    #grad_y = (y_next -  y_prev) / (2.*y.delta_x)
    grad_n = (n_next -  n) / (dr)
    dndr_full=interpax.Interpolator1D(r_grid_half[:-1],grad_n[:-1])(r_grid)
    #SET gradients to zero at radial coordinate r=0 
    dndr_full = dndr_full.at[0].set(0.)
    dndr_full = dndr_full.at[-1].set((4.*n.at[-2].get()-n.at[-3].get()-3*n_edge)/(-2*dr))   
    return dndr_full

@jit
def get_gradient_Er(y,dr):
    y_next = jnp.roll(y, shift=-1)
    #y_prev = jnp.roll(y shift=1)    
    #grad_y = (y_next -  y_prev) / (2.*y.delta_x)
    #grad_y = (y_next -  y_prev) / (y.delta_x)
    grad_y = (y_next -  y) / (dr)
    # Dirichlet boundary condition
    #grad_y = grad_y.at[0].set(0)
    #grad_edge=(4.*y.vals.at[-2].get()-y.vals.at[-3].get()-3*y.vals.at[-1].get())/(-2*y.delta_x)
    #grad_y = grad_y.at[-1].set(grad_edge)
    #grad_y = grad_y.at[-1].set(0)
    return grad_y


@jit
def gradient_no(y,dr):
    y_next = jnp.roll(y, shift=-1)
    diff = (y_next-y)/(dr)
    # Dirichlet boundary condition
    grad_y=jnp.roll(diff, shift=1)
    grad_y = grad_y.at[0].set(0)
    #grad_edge=(4.*y.vals.at[-2].get()-y.vals.at[-3].get()-3*y.vals.at[-1].get()+half_flux)/(-2*y.delta_x)
    #grad_y = grad_y.at[-1].set(grad_edge)
    grad_y = grad_y.at[-1].set(0)
    return grad_y


@jit
def get_diffusion_Er(y,r_grid,r_grid_half,Vprime_half,overVprime,dr):
    #Auxiliary Spatial discretizations for convective terms (this is missing the last term of turbulent fluxes)
    #Derivatives for electric field
    #aux=-Er.vals/(rho_grid*a_b)
    #aux=aux.at[0].set(0.)
    #aux2=Er.vals/jnp.square(rho_grid*a_b)
    #aux2=aux2.at[0].set(0.)
    #aux3=gradient_Er_original(Er).vals/(rho_grid*a_b)
    #aux3=aux3.at[0].set(0.)    
    #diffusion=Vpp_Vp*(aux+gradient_Er_original(Er).vals)+aux2-aux3+laplacian(Er).vals#+overVprime*gradient(gradient(Er)).vals+Vpp_Vp*(aux+gradient(Er).vals)
    #FluxEr=Vprime_half*rho_grid_half*a_b*(gradient_Er(SpatialDiscretisation(r0,r_final,-aux)).vals)#-Er_half/(rho_grid_half*a_b))
    #gradEr=gradient_Er(Er).vals
    #gradEr_full=interpax.Interpolator1D(r_grid_half[:-1],gradEr.vals[:-1])(r_grid)
    #Flux_Er_full=gradEr-Er_half/r_grid_half
    #Flux_Er_half=Vprime_half*interpax.Interpolator1D(r_grid_half,Flux_Er_full)(r_grid_half)
    y_half=interpax.Interpolator1D(r_grid,y,extrap=True)(r_grid_half)
    FluxEr=Vprime_half*(get_gradient_Er(y,dr)-y_half/r_grid_half)
    diffusion=gradient_no(FluxEr,dr)*overVprime
    diffusion=diffusion.at[0].set(0.0)
    diffusion=diffusion.at[-1].set(0.0)
    #diffusion=diffusion_Er(Er).vals/Vprime
    return diffusion


#Get thermodynamical forces
@jit
def get_Thermodynamical_Forces_A1(q,n,T,dndr,dTdr,Er):
    A1=dndr/n-1.5*dTdr/T-1.e+3*Er*q/(T*elementary_charge)    
    return A1

#Get thermodynamical forces
@jit
def get_Thermodynamical_Forces_A2(T,dTdr):
    A2=dTdr/T
    return A2

#Get thermodynamical forces
@jit
def get_Thermodynamical_Forces_A3(Er):
    A3=Er*0.      
    return A3


@jit
class Species(eqx.Module):
    number_species: int
    radial_points: int
    species_indeces: int
    mass_mp: Float[Array, "number_species"]  
    charge_qp: Float[Array, "number_species"]  
    temperature: ndarray # Float[Array, "number_species x radial_points"]    # in units of eV
    density: ndarray#Float[Array, "species x radial_points"]    # in units of particles/m^3
    Er: Float[Array, "radial_points"] 
    r_grid: Float[Array, "radial_points"] 
    r_grid_half: Float[Array, "radial_points"] 
    dr: float
    Vprime_half: Float[Array, "radial_points"] 
    overVprime: Float[Array, "radial_points"] 
    n_edge: float
    T_edge: float

    #v_thermal: Float[Array, "species x radial_points"]    # in units of particles/m^3
    @property
    def v_thermal(self):
        return jax.vmap(jax.vmap(get_v_thermal,in_axes=(None,0)),in_axes=(0,0))(self.mass,self.temperature)
    @property
    def charge(self):
        return self.charge_qp*elementary_charge
    @property
    def mass(self):
        return self.mass_mp*proton_mass
    @property
    def dTdr(self): 
        return jax.vmap(get_gradient_temperature,in_axes=(0,None,None,None,0))(self.temperature,self.r_grid,self.r_grid_half,self.dr,self.T_edge)
    @property
    def dndr(self): 
        return jax.vmap(get_gradient_density,in_axes=(0,None,None,None,0))(self.density,self.r_grid,self.r_grid_half,self.dr,self.n_edge)
    @property
    def dErdr(self): 
        return get_gradient_Er(self.Er) 
    @property
    def A1(self):
        return jax.vmap(get_Thermodynamical_Forces_A1,in_axes=(0,0,0,0,0,None))(self.charge,self.density,self.temperature,self.dndr,self.dTdr,self.Er)
    @property
    def A2(self):
        return jax.vmap(get_Thermodynamical_Forces_A2,in_axes=(0,0))(self.temperature,self.dTdr)
    @property
    def A3(self):
        return get_Thermodynamical_Forces_A3(self.Er)
    @property
    def diffusion_Er(self):
        return get_diffusion_Er(self.Er,self.r_grid,self.r_grid_half,self.Vprime_half,self.overVprime,self.dr)

def collisionality(species_a: int,  species: Species,v: float, r_index: int) -> float:
    """Collisionality between species a and others.

    Parameters
    ----------
    maxwellian_a : LocalMaxwellian
        Distribution function of primary species.
    v : float
        Speed being considered.
    *others : LocalMaxwellian
        Distribution functions for background species colliding with primary.

    Returns
    -------
    nu_a : float
        Collisionality of species a against background of others, in units of 1/s
    """
    nu = 0.0
    #I think we can do jnp.sum(jnp.vmap) here but let see with the normal one first TODO
    #for ma in range(species.number_species):
    #    nu += nuD_ab(species,species_a, ma, v, r_index)
    nu=jnp.sum(jax.vmap(nuD_ab,in_axes=(None,None,0,None,None))(species,species_a,species.species_indeces,v,r_index),axis=0)
    return nu


def nuD_ab(species: Species,species_a: int,species_b: int, v: float, r_index:int) -> float:
    """Pairwise collision freq. for species a colliding with species b at velocity v.

    Parameters
    ----------
    species_a : int
        index of species a in global class species
    species_b : int
        index of species b in global class species
    v : float
        Speed being considered.
    Species: species class

    Returns
    -------
    nu_ab : float
        Collisionality of species a against background of b, in units of 1/s

    """
    nb = species.density[species_b,r_index]
    vtb = species.v_thermal[species_b,r_index]
    prefactor = gamma_ab(species,species_a, species_b, v,r_index) * nb / v**3
    erf_part = jax.scipy.special.erf(v / vtb) - chandrasekhar(v / vtb)
    return prefactor * erf_part


def gamma_ab(species: Species,species_a: int,species_b: int, v: float, r_index: int) -> float:
    """Prefactor for pairwise collisionality."""
    lnlambda = coulomb_logarithm(species,species_a, species_b,r_index)
    ea, eb = species.charge[species_a], species.charge[species_b]
    ma = species.mass[species_a]
    Ta=species.temperature[species_a,r_index]
    return ea**2 * eb**2 * lnlambda / (4 * jnp.pi * epsilon_0**2 * ma**2)


def nupar_ab(species: Species,species_a: int, species_b: int, v: float, r_index: int) -> float:
    """Parallel collisionality."""
    nb = species.density[species_b,r_index]
    vtb = species.v_thermal[species_b,r_index]
    return (
        2 * gamma_ab(species,species_a, species_b, v,r_index) * nb / v**3 * chandrasekhar(v / vtb)
    )


def coulomb_logarithm(species: Species,species_a: int, species_b: int, r_index: int) -> float:
    """Coulomb logarithm for collisions between species a and b.
    Parameters
    ----------
    maxwellian_a : LocalMaxwellian
        Distribution function of primary species.
    maxwellian_b : LocalMaxwellian
        Distribution function of background species.
    Returns
    -------
    log(lambda) : float
    """
    #bmin, bmax =   impact_parameter(species,species_a, species_b,  r_index)
    #return jnp.log(bmax / bmin)
    #lnL = 25.3 + 1.15*jnp.log10(species.temperature[0,r_index]**2/species.density[0,r_index])  
    lnL = 32.2 + 1.15*jnp.log10(species.temperature[0,r_index]**2/species.density[0,r_index]) 
    #32.2+1.15*alog10(temp(1)**2/density(1))
    return lnL


def impact_parameter(species: Species,species_a: int, species_b: int, r_index: int) -> float:
    """Impact parameters for classical Coulomb collision."""
    bmin = jnp.maximum(
        impact_parameter_perp(species,species_a, species_b,r_index),
        debroglie_length(species,species_a, species_b,r_index),
    )
    bmax = debye_length(species,r_index)
    return bmin, bmax


def impact_parameter_perp(species: Species,species_a: int, species_b: int, r_index: int) -> float:
    """Distance of the closest approach for a 90° Coulomb collision."""
    m_reduced = (
        species.mass[species_a]
        * species.mass[species_b]
        / (species.mass[species_a] + species.mass[species_b])
    )
    v_th = jnp.sqrt(species.v_thermal[species_a,r_index] * species.v_thermal[species_b,r_index])
    return (
        species.charge[species_a]
        * species.charge[species_a]
        / (4 * jnp.pi * epsilon_0 * m_reduced * v_th**2)
    )


def debroglie_length(species: Species,species_a: int, species_b: int, r_index: int) -> float:
    """Thermal DeBroglie wavelength."""
    m_reduced = (
        species.mass[species_a]
        * species.mass[species_b]
        / (species.mass[species_a] + species.mass[species_b])
    )
    v_th = jnp.sqrt(species.v_thermal[species_a,r_index] * species.v_thermal[species_b,r_index])
    return hbar / (2 * m_reduced * v_th)


def debye_length(species: Species, r_index: int) -> float:
    """Scale length for charge screening."""
    den = 0
    for m in range(species.number_species):
        den += species.density[m,r_index] / (species.temperature[m,r_index] * JOULE_PER_EV) * species.charge[m]**2
    #den=jnp.sum(species.density[:,r_index] / (species.temperature[:,r_index] * JOULE_PER_EV) * species.charge[:]**2)
    return jnp.sqrt(epsilon_0 / den)


def chandrasekhar(x: jax.Array) -> jax.Array:
    """Chandrasekhar function."""
    return (
        jax.scipy.special.erf(x) - 2 * x / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2))
    ) / (2 * x**2)


def _dchandrasekhar(x):
    return 2 / jnp.sqrt(jnp.pi) * jnp.exp(-(x**2)) - 2 / x * chandrasekhar(x)
