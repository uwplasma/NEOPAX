import h5py as h5 
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit
from netCDF4 import Dataset
import interpax 
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx

class Field(eqx.Module):
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
        rho_half : Float[Array,'...'],
        rho_full : Float[Array,'...'],
        volume_p : float ,
        vp : Float[Array,'...'],
        iotaf : Float[Array,'...'],
        Psia : float,
        bmnc_b : Float[Array,'...'],
        rmnc_b : Float[Array,'...'],
        gmnc_b : Float[Array,'...'],
        xm_b : int,
        xn_b : int,
        bvco : Float[Array,'...'],
        buco : Float[Array,'...'],
    ):


        self.R0=rmnc_b[-1,0]
        self.a_b=np.sqrt(volume_p/(2*jnp.pi**2*self.R0))
        self.n_r=n_r
        self.rho_grid=jnp.linspace(0., 1., self.n_r)
        self.rho_grid_half=jnp.linspace((self.rho_grid.at[0].get()+self.rho_grid.at[1].get())*0.5, (self.rho_grid.at[0].get()+self.rho_grid.at[1].get())*0.5+self.rho_grid.at[-1].get(), self.n_r)
        self.r_grid=self.rho_grid*self.a_b
        self.r_grid_half=self.rho_grid_half*self.a_b
        self.dr=self.r_grid[2]-self.r_grid[1]

        for l in range(len(xm_b)):
            if(xm_b[l]==0 and xn_b[l]==0):
                B00=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)
                R00=interpax.Interpolator1D(rho_full[1:],rmnc_b[:,l],extrap=True)
                sqrtg00=interpax.Interpolator1D(rho_half[1:],gmnc_b[:,l],extrap=True)        
            if(xm_b[l]==1 and xn_b[l]==0):
                B10=interpax.Interpolator1D(rho_half[1:],bmnc_b[:,l],extrap=True)


        dVdr=interpax.Interpolator1D(rho_half[1:],vp[1:],extrap=True)
        self.Vprime=dVdr(self.rho_grid)*2.*self.rho_grid/self.a_b#*(2.*jnp.pi)**2
        self.Vprime_half=dVdr(self.rho_grid_half)*2.*self.rho_grid_half/self.a_b#*(2.*jnp.pi)**2
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
        self.sqrtg00_value=sqrtg00(self.rho_grid_half)
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




