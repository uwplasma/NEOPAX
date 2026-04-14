"""
Modular geometry models for NEOPAX.
- Registry-based selection (like transport_flux_models.py)
- JIT-compatible, differentiable (JAX-friendly)
"""
import dataclasses
import jax
import jax.numpy as jnp
from typing import Callable, Any
from jaxtyping import Array, Float, Int  
import functools

# Registry for geometry models
GEOMETRY_MODEL_REGISTRY: dict[str, Callable[..., "GeometryModelBase"]] = {}

def register_geometry_model(name: str, builder: Callable[..., "GeometryModelBase"]) -> None:
    GEOMETRY_MODEL_REGISTRY[str(name).strip().lower()] = builder

def get_geometry_model(name: str, **kwargs) -> "GeometryModelBase":
    key = str(name).strip().lower()
    if key not in GEOMETRY_MODEL_REGISTRY:
        raise ValueError(f"Unknown geometry model '{name}'.")
    return GEOMETRY_MODEL_REGISTRY[key](**kwargs)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class GeometryModelBase:
    """Abstract base class for geometry models."""
    def as_field(self):
        raise NotImplementedError

# --- VMEC/BOOZ geometry model ---
from netCDF4 import Dataset
import numpy as np
import interpax


@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class VmecBoozer(GeometryModelBase):
    n_r: int  #int = eqx.field(static=True)
    a_b: float
    Psia_value: float
    rho_grid: Float[Array, "n_r"]
    rho_grid_half: Float[Array, "n_r"]
    r_grid: Float[Array, "n_r"]
    r_grid_half: Float[Array, "n_r"]   
    full_grid_indices : Int[Array, "n_r"]   
    dr: float         
    Vprime: Float[Array, "n_r"]
    Vprime_half: Float[Array, "n_r"]
    overVprime: Float[Array, "n_r"]
    epsilon_t: Float[Array, "n_r"]
    B0: Float[Array, "n_r"]
    B_10: Float[Array, "n_r"]    
    enlogation: Float[Array, "n_r"] 
    iota: Float[Array, "n_r"]      
    R0: Float[Array, "n_r"]  
    B0prime: Float[Array, "n_r"]  
    curvature: Float[Array, "n_r"]      
    G_PS: Float[Array, "n_r"]  
    sqrtg00_value: Float[Array, "n_r"]            
    Bsqav: Float[Array, "n_r"]        
    I_value: Float[Array, "n_r"]        
    G_value: Float[Array, "n_r"]    

    


    def __init__(self, n_r, vmec=None, booz=None, **kwargs):
        # Support JAX pytree/dataclass reconstruction from all fields as kwargs
        field_names = [f.name for f in dataclasses.fields(type(self))]
        payload_fields = [name for name in field_names if name != "n_r"]
        if all(name in kwargs for name in payload_fields):
            for name in payload_fields:
                setattr(self, name, kwargs[name])
            self.n_r = n_r
            return

        # Default initialization from files
        if vmec is None or booz is None:
            raise ValueError("Must provide vmec and booz file paths for default initialization.")
        vfile = Dataset(vmec, mode="r")
        bfile = Dataset(booz, mode="r")
        ns = vfile.variables["ns"][:].filled()
        s_full = jnp.linspace(0,1,ns)
        s_half_list = [(i-0.5)/(ns-1) for i in range(0,ns)]
        s_half = jnp.array(s_half_list)
        rho_half = jnp.sqrt(s_half)
        rho_full = jnp.sqrt(s_full)
        volume_p = vfile.variables["volume_p"][:].filled()
        vp = vfile.variables["vp"][:].filled()
        iotaf = vfile.variables["iotaf"][:].filled()
        phi = vfile.variables["phi"][:].filled()
        Psia = jnp.abs(phi[-1])
        bmnc_b = bfile.variables["bmnc_b"][:].filled()
        rmnc_b = bfile.variables["rmnc_b"][:].filled()
        gmnc_b = bfile.variables['gmn_b'][:].filled()
        xm_b = bfile.variables['ixm_b'][:].filled()
        xn_b = bfile.variables['ixn_b'][:].filled()
        bvco = bfile.variables['bvco_b'][:].filled()
        buco = bfile.variables['buco_b'][:].filled()
        vfile.close()
        bfile.close()

        self.n_r = n_r
        self.full_grid_indices=jnp.arange(self.n_r) 


        self.R0 = rmnc_b[-1,0]
        self.a_b = np.sqrt(volume_p/(2*jnp.pi**2*self.R0))
        self.rho_grid = jnp.linspace(0., 1., self.n_r)
        self.rho_grid_half = jnp.concatenate([
            jnp.array([0.]),
            0.5 * (self.rho_grid[:-1] + self.rho_grid[1:]),
            jnp.array([1.])
        ])
        self.r_grid = self.rho_grid * self.a_b
        self.r_grid_half = self.rho_grid_half * self.a_b
        self.dr = self.r_grid[1] - self.r_grid[0]

        # Interpolators for equilibrium quantities
        for l in range(len(xm_b)):
            if(xm_b[l]==0 and xn_b[l]==0):
                B00 = interpax.Interpolator1D(rho_half[1:], bmnc_b[:,l], extrap=True)
                R00 = interpax.Interpolator1D(rho_full[1:], rmnc_b[:,l], extrap=True)
                sqrtg00 = interpax.Interpolator1D(rho_half[1:], gmnc_b[:,l], extrap=True)
            if(xm_b[l]==1 and xn_b[l]==0):
                B10 = interpax.Interpolator1D(rho_half[1:], bmnc_b[:,l], extrap=True)

        dVdr = interpax.Interpolator1D(rho_half[1:], vp[1:], extrap=True)
        self.Vprime = dVdr(self.rho_grid)*2.*self.rho_grid/self.a_b
        self.Vprime_half = dVdr(self.rho_grid_half)*2.*self.rho_grid_half/self.a_b
        self.overVprime = 1./self.Vprime
        self.overVprime = self.overVprime.at[0].set(0.0)

        iota_interp = interpax.Interpolator1D(rho_full[:], iotaf[:], extrap=True)
        self.iota = iota_interp(self.rho_grid)
        self.epsilon_t = self.rho_grid*self.a_b/R00(self.rho_grid)
        B_00 = B00(self.rho_grid)
        self.B_10 = B10(self.rho_grid)/B_00
        self.B0prime = jax.vmap(jax.grad(lambda r : B00(r)), in_axes=0)(self.r_grid)
        self.B0 = B00(self.r_grid)
        self.curvature = jnp.absolute(self.B_10)/self.epsilon_t
        self.curvature = self.curvature.at[0].set(0.0)
        self.enlogation = jnp.square(self.epsilon_t/self.B_10)

        G = interpax.Interpolator1D(rho_half[1:], bvco[1:], extrap=True)
        I = interpax.Interpolator1D(rho_half[1:], buco[1:], extrap=True)

        d0 = 4./3.
        d1 = 3.4229
        d2 = -2.5766
        d3 = -0.6039
        self.G_PS = 1.5*(d0*jnp.power(self.curvature/self.iota,2)*
                (1.+d1*jnp.power(self.epsilon_t,3.6)*(1.+d2*jnp.power(self.iota,1.6))
                +d3*jnp.power(self.epsilon_t,2)*(1.-jnp.power(self.curvature,2))))

        self.I_value = I(self.rho_grid)
        self.G_value = G(self.rho_grid)
        self.Psia_value = Psia
        self.sqrtg00_value = sqrtg00(self.rho_grid)
        self.Bsqav = (self.G_value+self.iota*self.I_value)/self.sqrtg00_value/jnp.power(self.B0,2)




register_geometry_model("default", VmecBoozer)
register_geometry_model("vmec_booz", VmecBoozer)

def extend_grid_with_ghosts(grid):
    """Extend a 1D grid array with one ghost cell on each side (extrapolated)."""
    dr = grid[1] - grid[0]
    left_ghost = grid[0] - dr
    right_ghost = grid[-1] + dr
    return jnp.concatenate([jnp.array([left_ghost]), grid, jnp.array([right_ghost])])

def extend_faces_with_ghosts(grid_half):
    """Extend a 1D face grid array for N+1 faces to N+3 faces (ghost faces)."""
    dr = grid_half[1] - grid_half[0]
    left_ghost = grid_half[0] - dr
    right_ghost = grid_half[-1] + dr
    return jnp.concatenate([jnp.array([left_ghost]), grid_half, jnp.array([right_ghost])])


