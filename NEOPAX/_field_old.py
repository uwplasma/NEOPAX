import h5py as h5 
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from netCDF4 import Dataset
import interpax 
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import dataclasses

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

@jax.tree_util.register_dataclass
@dataclasses.dataclass(eq=False)
class Field:
    """Magnetic field and config parameters.

    """

    # note: assumes (psi, theta, zeta) coordinates, not (rho, theta, zeta)
    n_r: int  #int = eqx.field(static=True)
    a_b: float
    Psia_value: float
    rho_grid: Float[Array, "n_r"]
    rho_grid_half: Float[Array, "n_r"]
    r_grid: Float[Array, "n_r"]
    r_grid_half: Float[Array, "n_r"]   
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
    

    def __init__(
        self,
        n_r: int,
        rho_half : Float[Array,'...']=None,
        rho_full : Float[Array,'...']=None,
        volume_p : float=None,
        vp : Float[Array,'...']=None,
        iotaf : Float[Array,'...']=None,
        Psia : float=None,
        bmnc_b : Float[Array,'...']=None,
        rmnc_b : Float[Array,'...']=None,
        gmnc_b : Float[Array,'...']=None,
        xm_b : int=None,
        xn_b : int=None,
        bvco : Float[Array,'...']=None,
        buco : Float[Array,'...']=None,
        **kwargs,
    ):

        self.n_r = n_r

        # JAX dataclass pytree reconstruction may call Field(...) with all
        # dataclass fields as keyword payload.
        field_names = [f.name for f in dataclasses.fields(type(self))]
        payload_fields = [name for name in field_names if name != "n_r"]
        if all(name in kwargs for name in payload_fields):
            for name in payload_fields:
                setattr(self, name, kwargs[name])
            return


        self.R0=rmnc_b[-1,0]
        self.a_b=np.sqrt(volume_p/(2*jnp.pi**2*self.R0))
        self.n_r=n_r
        self.rho_grid = jnp.linspace(0., 1., self.n_r)
        # torax-style: n_r+1 faces, endpoints at 0 and 1, faces halfway between centers
        self.rho_grid_half = jnp.concatenate([
            jnp.array([0.]),
            0.5 * (self.rho_grid[:-1] + self.rho_grid[1:]),
            jnp.array([1.])
        ])
        self.r_grid = self.rho_grid * self.a_b
        self.r_grid_half = self.rho_grid_half * self.a_b
        self.dr = self.r_grid[1] - self.r_grid[0]

        # --- Add ghost cells to r_grid and r_grid_half ---
        self.r_grid_full_ghost = extend_grid_with_ghosts(self.r_grid)
        self.r_grid_half_ghost = extend_faces_with_ghosts(self.r_grid_half)
        # Now r_grid_full_ghost has n_r+2 points, r_grid_half_ghost has n_r+3 faces

        for l in range(len(xm_b)):
            if(xm_b[l]==0 and xn_b[l]==0):
                B00=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)
                R00=interpax.Interpolator1D(rho_full[1:],rmnc_b[:,l],extrap=True)
                sqrtg00=interpax.Interpolator1D(rho_half[1:],gmnc_b[:,l],extrap=True)        
            if(xm_b[l]==1 and xn_b[l]==0):
                B10=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)


        # VMEC stores `vp = dV/ds` without the conventional 4*pi^2 factor.
        volume_scale = (2.0 * jnp.pi) ** 2
        dVdr=interpax.Interpolator1D(rho_half[1:],vp[1:],extrap=True)
        self.Vprime=dVdr(self.rho_grid)*2.*self.rho_grid/self.a_b*volume_scale
        self.Vprime_half=dVdr(self.rho_grid_half)*2.*self.rho_grid_half/self.a_b*volume_scale
        self.overVprime=1./self.Vprime
        self.overVprime=self.overVprime.at[0].set(0.0)

        iota=interpax.Interpolator1D(rho_full[:],iotaf[:],extrap=True)
        self.iota=iota(self.rho_grid)
        self.epsilon_t=self.rho_grid*self.a_b/R00(self.rho_grid)
        B_00=B00(self.rho_grid)
        self.B_10=B10(self.rho_grid)/B_00
        self.B0prime=jax.vmap(jax.grad(lambda r : B00(r)), in_axes=0)(self.r_grid)
        self.B0=B00(self.r_grid)
        self.curvature=jnp.absolute(self.B_10)/self.epsilon_t
        self.curvature=self.curvature.at[0].set(0.0)
        self.enlogation=jnp.square(self.epsilon_t/self.B_10)

 
        G=interpax.Interpolator1D(rho_half[1:],bvco[1:],extrap=True)
        I=interpax.Interpolator1D(rho_half[1:],buco[1:],extrap=True)

        #Coefficients for PS factor
        d0=4./3.
        d1=3.4229
        d2=-2.5766
        d3=-0.6039
        self.G_PS=1.5*(d0*jnp.power(self.curvature/self.iota,2)*
                (1.+d1*jnp.power(self.epsilon_t,3.6)*(1.+d2*jnp.power(self.iota,1.6))
                +d3*jnp.power(self.epsilon_t,2)*(1.-jnp.power(self.curvature,2))))

        self.I_value=I(self.rho_grid)
        self.G_value=G(self.rho_grid)
        self.Psia_value=Psia
        self.sqrtg00_value=sqrtg00(self.rho_grid)
        #Bsqav=jnp.power(2.*jnp.pi,2)*(G(rho_grid)+iota(rho_grid)*I(rho_grid))/dVdr(rho_grid)
        self.Bsqav=(self.G_value+self.iota*self.I_value)/self.sqrtg00_value/jnp.power(self.B0,2)
        #Important geometrical quantities interpolated from equilibrium
 


    @classmethod
    def read_vmec_booz(cls,
        n_r,
        vmec,
        booz,
    ):
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        vmec : path-like
            Path to vmec wout file.
        booz : path-like
            Path to booz_xform file.
        """
        from netCDF4 import Dataset

        #This should go to equilibrium reader
        vfile = Dataset(vmec, mode="r")
        bfile = Dataset(booz, mode="r")

        ns = vfile.variables["ns"][:].filled()
        s_full = jnp.linspace(0,1,ns)  #This is s_full
        s_half_list = [(i-0.5)/(ns-1) for i in range(0,ns)] #This is s_half
        s_half =jnp.array(s_half_list)

        rho_half=jnp.sqrt(s_half)
        rho_full=jnp.sqrt(s_full)

        #Vprime = vfile.variables["vp"][:].filled()
        #Aminor_p = vfile.variables["Aminor_p"][:].filled()   
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

        vfile.close()
        bfile.close()

        data = {}
        data["rho_half"] = rho_half
        data["rho_full"] = rho_full
        data["volume_p"] = volume_p
        data["vp"] = vp
        data["iotaf"] = iotaf
        data["Psia"] = Psia
        data["bmnc_b"] = bmnc_b
        data["rmnc_b"] = rmnc_b
        data["gmnc_b"] = gmnc_b
        data["xm_b"] = xm_b
        data["xn_b"] = xn_b
        data["bvco"] = bvco
        data["buco"] = buco

        return cls(n_r,**data)




