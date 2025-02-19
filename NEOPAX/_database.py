import h5py as h5
import jax
import jax.numpy as jnp
from _field import a_b
from scipy.constants import elementary_charge
from _parameters import monkes_file


#Read monkes database
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
#Er_list=Er[0]#jnp.log10(np.maximum(1.e-8,jnp.abs(Er[0])))

D11_log=jnp.log10(D11)
nu_log=jnp.log10(nu_v)
D13=jnp.array(D13)
D33=jnp.array(D33)
#nu_t = 10**jnp.linspace(np.log10(1.e-9),np.log10(1000.), 101)
#Er_t = jnp.linspace(Er.min(), Er.max(), 101)
#rho_t = jnp.linspace(0.,1., 101)
#Er_pm = jnp.linspace(-50., 50., 101)
#rho_pm = jnp.linspace(0., 1.0, 21)

#Create interpolation object of monoenergetic database
#Create interpolation object of monoenergetic database
#monodata=interpax.Interpolator3D(rho*a_b,nu_log,Er_list,D11_log,extrap=True)



