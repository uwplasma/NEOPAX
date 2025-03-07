import h5py as h5
import jax
import jax.numpy as jnp
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jaxtyping import Array, Float  # https://github.com/google/jaxtyping
import equinox as eqx
import interpax


#Monoenergetic database class
class Monoenergetic(eqx.Module):
    """Monoenergetic database.

    """

    # note: assumes (psi, theta, zeta) coordinates, not (rho, theta, zeta)
    a_b: float  #int = eqx.field(static=True)
    D11_lower_limit: float
    Er_lower_limit: float
    Er_lower_limit_log : float
    low_limit_r : float
    r1_lim : float
    rmn2_lim: float 
    r1 : float
    r2: float
    r3: float
    rnm3: float
    rnm2 : float
    rnm1: float
    rho: Float[Array,'...']
    nu_log : Float[Array,'...'] 
    Er_list: Float[Array,'...']
    D11_log: Float[Array,'...']
    D13 : Float[Array,'...'] 
    D33 : Float[Array,'...']

    def __init__(
        self,
        a_b: float,
        rho: Float[Array,'...'],
        nu_log : Float[Array,'...'], 
        Er_list: Float[Array,'...'],
        D11_log: Float[Array,'...'],
        D13 : Float[Array,'...'],
        D33 : Float[Array,'...'],
    ):

        self.a_b=a_b
        self.rho=rho
        self.nu_log=nu_log
        self.Er_list=Er_list
        self.D11_log=D11_log
        self.D13=D13
        self.D33=D33
        self.D11_lower_limit=jnp.array(-12.0)
        self.Er_lower_limit=1.e-8
        self.Er_lower_limit_log=jnp.log10(1.e-8)
        self.low_limit_r=jnp.array(1.e-3*self.a_b)
        self.r1_lim=self.a_b*self.rho[1]
        self.rmn2_lim=self.a_b*self.rho[-2]
        self.r1=self.rho[0]*self.a_b
        self.r2=self.rho[1]*self.a_b
        self.r3=self.rho[2]*self.a_b
        self.rnm3=self.rho[-3]*self.a_b
        self.rnm2=self.rho[-2]*self.a_b
        self.rnm1=self.rho[-1]*self.a_b


    @classmethod
    def read_monkes(cls,
        a_b,                    
        monkes_file,
    ):
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        monkes_file : path-like
            Path to vmec wout file.
        """

        file=h5.File(monkes_file ,'r')
        D11=file['D11'][()]
        D13=file['D13'][()]
        D33=file['D33'][()]
        nu_v=file['nu_v'][()]
        Er_save=file['Er'][()]
        drds=file['drds'][()]
        rho=jnp.array(file['rho'][()])
        Er=file['Er'][()]
        Es=file['Es'][()]
        Er_tilde=file['Er_tilde'][()]
        Er_to_Ertilde=file['Er_to_Ertilde'][()]
        file.close()

        Er_ref=jnp.array(1.e-8)

        Er_list=jnp.zeros((len(rho),len(Er_tilde)))
        for j in range(len(rho)):
            D11[j,:,:]=D11[j,:,:]*jnp.power(drds[j],2)
            D13[j,:,:]=D13[j,:,:]*drds[j]
            for k in range(len(Er_tilde)):
                Er_list=Er_list.at[j,k].set(jnp.log10(jnp.maximum(1.e-8,jnp.abs(Er[0,k])/(a_b*rho.at[j].get()))))
                D33[j,:,k]=D33[j,:,k]*nu_v  #Theres a B0^2*B^2_flux_average in NTSS TODO, probably not necessary 
        #Er_list=jnp.log10(jnp.maximum(1.e-8,jnp.abs(Er[0])))

        D11_log=jnp.log10(D11)
        nu_log=jnp.log10(nu_v)
        D13=jnp.array(D13)
        D33=jnp.array(D33)

        data = {}
        data["a_b"]=a_b
        data["rho"] = rho
        data["nu_log"] = nu_log
        data["Er_list"] = Er_list
        data["D11_log"] = D11_log
        data["D13"] = D13
        data["D33"] = D33

        return cls(**data)


    @classmethod
    def read_data(cls,
        a_b,  
        rho,                  
        nu_v,
        Er,
        drds,
        D11,
        D13,
        D33
    ):
        """Construct Field from BOOZ_XFORM file.

        Parameters
        ----------
        monkes_file : path-like
            Path to vmec wout file.
        """

        Er_ref=jnp.array(1.e-8)

        Er_list=jnp.zeros((len(rho),len(Er.shape[2])))
        for j in range(len(rho)):
            D11[j,:,:]=D11[j,:,:]*jnp.power(drds[j],2)
            D13[j,:,:]=D13[j,:,:]*drds[j]
            for k in range(Er.shape[2]):
                Er_list=Er_list.at[j,k].set(jnp.log10(jnp.maximum(1.e-8,jnp.abs(Er[0,k])/(a_b*rho.at[j].get()))))
                D33[j,:,k]=D33[j,:,k]*nu_v  #Theres a B0^2*B^2_flux_average in NTSS TODO, probably not necessary 


        D11_log=jnp.log10(D11)
        nu_log=jnp.log10(nu_v)
        D13=jnp.array(D13)
        D33=jnp.array(D33)

        data = {}
        data["a_b"]=a_b
        data["rho"] = rho
        data["nu_log"] = nu_log
        data["Er_list"] = Er_list
        data["D11_log"] = D11_log
        data["D13"] = D13
        data["D33"] = D33

        return cls(**data)

#    @classmethod
#    def run_monkes(cls,
#        a_b,                    
#        vmec_file,
#        field_neopax
#    ):



        #Resolution parameters
    #    nt = 25
    #    nz = 25
        #Pitch angle resolution
    #    nl=60  

        #Typical rho, nu/v and E_rtilde/(v B0) values used in DKES IPP databases 
    #    rho=jnp.array([0.12247,0.25,0.375,0.5,0.625,0.75,0.875])        
    #    nu_v=jnp.array([1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2,3.e-2,1.e-1,3.e-1])
    #    Er_tilde=jnp.array([0.0,1.e-6,3.e-6,1.e-5,3.e-5,1.e-4,3.e-4,1.e-3,3.e-3,1.e-2])




        #Create arrays for Dij's monoenergetic scan data 
    #    D11=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))
    #    D13=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))
    #    D31=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))
    #    D33=jnp.zeros((len(rho),len(nu_v),len(Er_tilde)))
    #    B00_rho=interpax.Interpolator1D(field_neopax.rho_grid,field_neopax.B0,extrap=True)(rho)
    #    dr_tildedr=2.*field_neopax.Psia_value/(a_b**2*B00_rho)*a_b/(rho*2.)

        #Loop for every collisionality and electric field value to obtain the monoenergetic scan 
        #Use internal solve as we do not care for species information in the monoenergetic database
        #Using normal for loop here as it serves only for benchmark-> put into vmap to speed up in real calculations
    #    for si in range(len(rho)):
    #        field_monkes = monkes.Field.from_vmec_s(vmec_file, rho[si]**2, nt, nz)  
    #        for j in range(len(nu_v)):       
    #            for i in range(len(Er_tilde)):
    #                #Here we use input 
    #                Es[si,i]=Er_tilde[i]*dr_tildeds[si]*B00_rho[si] #Notice we need to multiply by B00 and dr_tildeds factors due to IPP DKES weird coordinates
    #                #Calculate one Dij matrix 
    #                Dij, f, s = monkes._core.monoenergetic_dke_solve_internal(field, nl=nl, Erhat=Es[si,i],nuhat=nu_v[j])
    #                D11[si,j,i]=Dij[0,0]
    #                D13[si,j,i]=Dij[0,2]
    #                D31[si,j,i]=Dij[2,0]
    #                D33[si,j,i]=Dij[2,2]    


    #    Er_ref=jnp.array(1.e-8)

    #    Er_list=jnp.zeros((len(rho),len(Er_tilde)))
    #    for j in range(len(rho)):
    #        D11[j,:,:]=D11[j,:,:]*jnp.power(drds[j],2)
    #        D13[j,:,:]=D13[j,:,:]*drds[j]
    #        for k in range(len(Er_tilde)):
    #            Er_list=Er_list.at[j,k].set(jnp.log10(jnp.maximum(1.e-8,jnp.abs(Er[0,k])/(a_b*rho.at[j].get()))))
    #            D33[j,:,k]=D33[j,:,k]*nu_v  #Theres a B0^2*B^2_flux_average in NTSS TODO, probably not necessary 
        #Er_list=jnp.log10(jnp.maximum(1.e-8,jnp.abs(Er[0])))

    #    D11_log=jnp.log10(D11)
    #    nu_log=jnp.log10(nu_v)
    #    D13=jnp.array(D13)
    #    D33=jnp.array(D33)

    #    data = {}
    #    data["a_b"]=a_b
    #    data["rho"] = rho
    #    data["nu_log"] = nu_log
    #    data["Er_list"] = Er_list
    #    data["D11_log"] = D11_log
    #    data["D13"] = D13
    #    data["D33"] = D33

    #    return cls(**data)


