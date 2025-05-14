import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import orthax
import equinox as eqx
from jaxtyping import Array, Float, Int  # https://github.com/google/jaxtyping


class Grid(eqx.Module):
    #Defining radial and energy grids
    n_r: int
    n_x: int
    n_species: int
    n_order: int
    Sonine_expansion : Float[Array, "n_order"]
    species_indeces : Int[Array, "n_species"]
    full_grid_indeces : Int[Array, "n_r"]    
    sonine_indeces : Int[Array, "n_order"] 
    x : Float[Array, "n_x"]      
    xWeights : Float[Array, "n_x"] 
    v_norm : Float[Array, "n_x"] 
    L11_weight : Float[Array, "n_x"] 
    L12_weight : Float[Array, "n_x"] 
    L22_weight : Float[Array, "n_x"] 
    L13_weight : Float[Array, "n_x"] 
    L23_weight : Float[Array, "n_x"] 
    L33_weight : Float[Array, "n_x"] 
    L24_weight : Float[Array, "n_x"] 
    L25_weight : Float[Array, "n_x"]                    
    L43_weight : Float[Array, "n_x"] 
    L44_weight : Float[Array, "n_x"] 
    L45_weight : Float[Array, "n_x"]   
    L55_weight : Float[Array, "n_x"]     
     
    def __init__(
        self,
        n_r: int,
        n_x: int,
        n_species: int,
    ):
        
        self.n_r=n_r
        self.n_x=n_x
        self.n_species=n_species
        self.n_order=3
        self.Sonine_expansion=jnp.array([1.0,0.4,8.0/35.0]) #order 2 Sonine expansion, fixed for now
        self.species_indeces=jnp.arange(self.n_species) 
        self.full_grid_indeces=jnp.arange(self.n_r) 
        self.sonine_indeces=jnp.arange(len(self.Sonine_expansion))
        xgrid=orthax.laguerre.laggauss(self.n_x)
        self.x=xgrid[0]
        self.xWeights=xgrid[1]
        self.v_norm=jnp.sqrt(self.x)
        self.L11_weight=jnp.power(self.x,2.)
        self.L12_weight=jnp.power(self.x,3.)
        self.L22_weight=jnp.power(self.x,4.)
        self.L13_weight=jnp.power(self.x,1.5)
        self.L23_weight=jnp.power(self.x,2.5)
        self.L33_weight=jnp.power(self.x,1.)        
        self.L24_weight=jnp.power(self.x,3.5)        
        self.L25_weight=jnp.power(self.x,4.5)   
        self.L43_weight=jnp.power(self.x,2.)        
        self.L44_weight=jnp.power(self.x,3.) 
        self.L45_weight=jnp.power(self.x,4.) 
        self.L55_weight=jnp.power(self.x,5.)
            
    @classmethod
    def create_standard(cls,n_r,n_x,n_species):
        return cls(n_r,n_x,n_species)
