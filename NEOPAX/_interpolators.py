import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import interpax
from _field import a_b
from _database import rho,nu_log,Er_list,D11_log,D13,D33

D11_lower_limit=jnp.array(-12.0)
Er_lower_limit=1.e-8
Er_lower_limit_log=jnp.log10(1.e-8)
low_limit_r=jnp.array(1.e-3*a_b)
r1_lim=jnp.array(a_b*rho[1])
rmn2_lim=jnp.array(a_b*rho[-2])
r1=rho[0]*a_b
r2=rho[1]*a_b
r3=rho[2]*a_b
rnm3=rho[-3]*a_b
rnm2=rho[-2]*a_b
rnm1=rho[-1]*a_b



def interpolator_nu_low_Er0(index,grid_nu):
    #return D11_log[index,0,0]-(nu_log[0]-grid_nu)*(D11_log[index,1,0]-D11_log[index,0,0])/(nu_log[1]-nu_log[0])
    return D11_log[index,0,0]+(nu_log[0]-grid_nu)#*(D11_log[index,1,0]-D11_log[index,0,0])/(nu_log[1]-nu_log[0])


def interpolator_nu_mid_Er0(index,grid_nu):
    return interpax.Interpolator1D(nu_log,D11_log[index,:,0],extrap=False)(grid_nu)


def interpolator_nu_large_Er0(index,grid_nu):
    #return D11_log[index,-1,0]+(grid_nu-nu_log[-1])*(D11_log[index,-1,0]-D11_log[index,-2,0])/(nu_log[-1]-nu_log[-2])
    return D11_log[index,-1,0]+(grid_nu-nu_log[-1])#*(D11_log[index,-1,0]-D11_log[index,-2,0])/(nu_log[-1]-nu_log[-2])

def interpolator_nu_low_Er_finite(index,grid_nu,grid_Er):
    #Calculate Er_h
    Er_h=(nu_log[0]-grid_nu)/3.0+grid_Er  #This works with grid_Er as log(Er)
    return jnp.select(condlist=[Er_h<Er_list[index,-1],Er_h>=Er_list[index,-1]],
                      choicelist=[interpax.Interpolator1D(Er_list[index,:],D11_log[index,0,:],extrap=False)(Er_h),D11_lower_limit],default=0)



def interpolator_nu_large_Er_finite(index,grid_nu,grid_Er):
    #This actually uses Tokamak + Er fit in NTSS!!! Have to update but for now I believe this is high collisionality so it will not affect much the comparison.
    # Thus mantaining general extrapolation 
    #TODO!! Create function for tokamak fit
    return interpax.Interpolator2D(nu_log,Er_list[index,:],D11_log[index,:,:],extrap=True)(grid_nu,grid_Er)



def interpolator_nu_mid_Er_finite_npoints3_low(index,grid_nu,grid_Er):
    #This is 3 points interpolation in nu
    #Calculate 3 points of nu for interpolatio
    nu0=nu_log.at[0].get()
    nu1=nu_log.at[1].get()
    nu2=nu_log.at[2].get()
    d11_nu=jnp.select(condlist=[grid_Er<=Er_list[index,0],
                     (grid_Er>Er_list[index,0])&(grid_Er<=Er_list[index,-1]),
                     grid_Er>Er_list[index,-1]],
                     choicelist=[jnp.array([D11_log.at[index,0,0].get(),D11_log.at[index,1,0].get(),D11_log.at[index,2,0].get()]),
                                  jnp.array([interpax.Interpolator1D(Er_list[index,:],D11_log[index,0,:],extrap=False)(grid_Er),
                                   interpax.Interpolator1D(Er_list[index,:],D11_log[index,1,:],extrap=False)(grid_Er),
                                   interpax.Interpolator1D(Er_list[index,:],D11_log[index,2,:],extrap=False)(grid_Er)]),
                                  jnp.array([D11_lower_limit,D11_lower_limit,D11_lower_limit])],default=0)
    d11_0=d11_nu.at[0].get()
    d11_1=d11_nu.at[1].get()
    d11_2=d11_nu.at[2].get()
    def nu_npoints3_interpolation_low(nu0,nu1,nu2,grid_nu,d11_0,d11_1,d11_2):
        h0 = (grid_nu-nu1)*(grid_nu-nu2)/((nu0-nu1)*(nu0-nu2))
        h1 = (grid_nu-nu0)*(grid_nu-nu2)/((nu1-nu0)*(nu1-nu2))
        h2 = (grid_nu-nu0)*(grid_nu-nu1)/((nu2-nu0)*(nu2-nu1))
        xg11 = h0*d11_0+h1*d11_1+h2*d11_2
        return xg11
    #return jnp.select(condlist=[(d11_2<=D11_lower_limit) or ((d11_1<=D11_lower_limit) & (grid_nu<= nu1)) 
    #                            or ((d11_0<=D11_lower_limit) & (grid_nu<= nu0)),
    #                            (d11_2>D11_lower_limit) and ((d11_1>D11_lower_limit) or (grid_nu> nu1)) 
    #                            and ((d11_0>D11_lower_limit) or (grid_nu> nu0))],
    #                  choicelist=[D11_lower_limit,nu_npoints3_interpolation_low(nu0,nu1,nu2,grid_nu,d11_0,d11_1,d11_2)],default=0)
    return nu_npoints3_interpolation_low(nu0,nu1,nu2,grid_nu,d11_0,d11_1,d11_2)

def interpolator_nu_mid_Er_finite_npoints3_high(index,grid_nu,grid_Er):
    #This is 3 points interpolation in nu
    #Calculate 3 points of nu for interpolatio
    nu0=nu_log.at[-3].get()
    nu1=nu_log.at[-2].get()
    nu2=nu_log.at[-1].get()
    d11_nu=jnp.select(condlist=[grid_Er<=Er_list[index,0],
                     (grid_Er>Er_list[index,0])&(grid_Er<=Er_list[index,-1]),
                     grid_Er>Er_list[index,-1] ],
                     choicelist=[jnp.array([D11_log.at[index,-3,0].get(),D11_log.at[index,-2,0].get(),D11_log.at[index,-1,0].get()]),
                                  jnp.array([interpax.Interpolator1D(Er_list[index,:],D11_log[index,-3,:],extrap=False)(grid_Er),
                                   interpax.Interpolator1D(Er_list[index,:],D11_log[index,-2,:],extrap=False)(grid_Er),
                                   interpax.Interpolator1D(Er_list[index,:],D11_log[index,-1,:],extrap=False)(grid_Er)]),
                                  jnp.array([D11_lower_limit,D11_lower_limit,D11_lower_limit])],default=0)
    d11_0=d11_nu.at[0].get()
    d11_1=d11_nu.at[1].get()
    d11_2=d11_nu.at[2].get()
    def nu_npoints3_interpolation_high(nu0,nu1,nu2,grid_nu,d11_0,d11_1,d11_2):
        h0 = (grid_nu-nu1)*(grid_nu-nu2)/((nu0-nu1)*(nu0-nu2))
        h1 = (grid_nu-nu0)*(grid_nu-nu2)/((nu1-nu0)*(nu1-nu2))
        h2 = (grid_nu-nu0)*(grid_nu-nu1)/((nu2-nu0)*(nu2-nu1))
        xg11 = h0*d11_0+h1*d11_1+h2*d11_2
        return xg11
#    return jnp.select(condlist=[(d11_2<=D11_lower_limit) or ((d11_1<=D11_lower_limit) & (grid_nu<= nu1)) 
#                                or ((d11_0<=D11_lower_limit) & (grid_nu<= nu0)),
#                                (d11_2>D11_lower_limit) and ((d11_1>D11_lower_limit) or (grid_nu> nu1)) 
#                                and ((d11_0>D11_lower_limit) or (grid_nu> nu0))],
#                      choicelist=[D11_lower_limit,nu_npoints3_interpolation_high(nu0,nu1,nu2,grid_nu,d11_0,d11_1,d11_2)],default=0)
    return nu_npoints3_interpolation_high(nu0,nu1,nu2,grid_nu,d11_0,d11_1,d11_2)


def interpolator_nu_mid_Er_finite_npoints4(index,grid_nu,grid_Er):
    #This is 4 points interpolation in nu
    #Calculate 4 points of nu for interpolation
    arr=grid_nu-nu_log[1:-1]
    index_nu = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf))+1
    idx0=index_nu-1
    idx1=index_nu-2
    idx2=index_nu
    idx3=index_nu+1
    nu0=nu_log.at[idx0].get()
    nu1=nu_log.at[idx1].get()
    nu2=nu_log.at[idx2].get()
    nu3=nu_log.at[idx3].get()
    d11_nu=jnp.select(condlist=[grid_Er<=Er_list[index,0],
                     (grid_Er>Er_list[index,0])&(grid_Er<=Er_list[index,-1]),
                     grid_Er>Er_list[index,-1] ],
                     choicelist=[jnp.array([D11_log.at[index,idx0,0].get(),D11_log.at[index,idx1,0].get(),D11_log.at[index,idx2,0].get(),D11_log.at[index,idx3,0].get()]),
                                  jnp.array([interpax.Interpolator1D(Er_list[index,:],D11_log[index,idx0,:],extrap=False)(grid_Er),
                                   interpax.Interpolator1D(Er_list[index,:],D11_log[index,idx1,:],extrap=False)(grid_Er),
                                   interpax.Interpolator1D(Er_list[index,:],D11_log[index,idx2,:],extrap=False)(grid_Er),
                                   interpax.Interpolator1D(Er_list[index,:],D11_log[index,idx3,:],extrap=False)(grid_Er)]),
                                  jnp.array([D11_lower_limit,D11_lower_limit,D11_lower_limit,D11_lower_limit])],default=0)
    d11_0=d11_nu.at[0].get()
    d11_1=d11_nu.at[1].get()
    d11_2=d11_nu.at[2].get()
    d11_3=d11_nu.at[3].get()
    def nu_npoints4_interpolation(nu0,nu1,nu2,nu3,grid_nu,d11_0,d11_1,d11_2,d11_3):
        dhl0 =(nu1-nu2)/((nu0-nu1)*(nu0-nu2))
        dhl1 = (2.*nu1-nu0-nu2)/((nu1-nu0)*(nu1-nu2))
        dhl2 = (nu1-nu0)/((nu2-nu0)*(nu2-nu1))
        dhu0 = (nu2-nu3)/((nu1-nu2)*(nu1-nu3))
        dhu1 = (2.*nu2-nu1-nu3)/((nu2-nu1)*(nu2-nu3))
        dhu2 = (nu2-nu1)/((nu3-nu1)*(nu3-nu2))
        dg11l = dhl0*d11_0+dhl1*d11_1+dhl2*d11_2
        dg11u = dhu0*d11_1+dhu1*d11_2+dhu2*d11_3
        dg11_new=jnp.select(condlist=[dg11l*dg11u <= 0.0,dg11l*dg11u>0],
                        choicelist=[0.2*jnp.array([dg11l,dg11u]),jnp.array([dg11l,dg11u])],default=0)
        dxnu21= nu2-nu1
        xnun  = (grid_nu-nu_log[idx1])/dxnu21
        ha1   = 3.*(d11_2-d11_1)-(2.*dg11_new[0]+dg11_new[1])*dxnu21
        hb1   =-2.*(d11_2-d11_1)+(dg11_new[0]+dg11_new[1])*dxnu21
        xg11 = d11_1+xnun*(dg11_new[0]*dxnu21+xnun*(ha1+xnun*hb1))
        return xg11
    return jnp.select(condlist=[d11_nu[2]<=D11_lower_limit,d11_nu[2]>D11_lower_limit],
                      choicelist=[D11_lower_limit,nu_npoints4_interpolation(nu0,nu1,nu2,nu3,grid_nu,d11_0,d11_1,d11_2,d11_3)],default=0)





def interpolator_nu_Er_general_NTSS(index,grid_nu,grid_Er):
    x11=jnp.select(condlist=[(grid_Er<=Er_lower_limit_log) & (grid_nu < nu_log[0]),   #nu_low Er_low
                                (grid_Er<=Er_lower_limit_log) & ((grid_nu >= nu_log[0]) & (grid_nu <= nu_log[-1])), #nu_mid, Er_low
                                (grid_Er<=Er_lower_limit_log) & (grid_nu > nu_log[-1]), #nu_high, Er_low
                                (grid_Er>Er_lower_limit_log) & (grid_nu < nu_log[0]),   #nu_low, Er_finite
                                (grid_Er>Er_lower_limit_log) & ((grid_nu >= nu_log[0]) & (grid_nu <= nu_log[1])),  #between nu[0] and nu[1], with Er finite, 3 points
                                (grid_Er>Er_lower_limit_log) & ((grid_nu >= nu_log[-2]) & (grid_nu <= nu_log[-1])),  #between nu[-2] and nu[-1], with Er finite, 3 points
                                (grid_Er>Er_lower_limit_log) & ((grid_nu > nu_log[1]) & (grid_nu < nu_log[-2])),  #between nu[1] and nu[2] (mid nu), with Er finite, 4 points
                                ###(grid_Er>Er_lower_limit) & ((grid_nu > nu_log[0]) & (grid_nu < nu_log[-1])),  #between nu[1] and nu[2] (mid nu), with Er finite, 4 points                                
                                (grid_Er>Er_lower_limit_log) & (grid_nu > nu_log[-1]) #nu_large, Er_finite
                                ],
                      choicelist=[interpolator_nu_low_Er0(index,grid_nu),
                                  interpolator_nu_mid_Er0(index,grid_nu),
                                  interpolator_nu_large_Er0(index,grid_nu),
                                  interpolator_nu_low_Er_finite(index,grid_nu,grid_Er),
                                  interpolator_nu_mid_Er_finite_npoints3_low(index,grid_nu,grid_Er),
                                  interpolator_nu_mid_Er_finite_npoints3_high(index,grid_nu,grid_Er),
                                  interpolator_nu_mid_Er_finite_npoints4(index,grid_nu,grid_Er),
                                  #interpolator_nu_mid_Er_finite(index,grid_nu,grid_Er),                                  
                                  interpolator_nu_large_Er_finite(index,grid_nu,grid_Er)],default=0)
                                ####interpax.Interpolator2D(nu_log,Er_list[index,:],D11_log[index,:,:],extrap=True)(grid_nu,grid_Er)],default=0)
    x13=interpax.Interpolator2D(nu_log,Er_list[index,:],D13[index,:,:],extrap=True)(grid_nu,grid_Er)
    x33=interpax.Interpolator2D(nu_log,Er_list[index,:],D33[index,:,:],extrap=True)(grid_nu,grid_Er)
    return x11,x13,x33                                





@jit
#####Interpolators, which should go to the _interpolators.py
def interpolator_nu_Er_general(index,grid_nu,grid_Er):
    x11=interpax.Interpolator2D(nu_log,Er_list[index,:],D11_log[index,:,:],extrap=True)(grid_nu,grid_Er)
    x13=interpax.Interpolator2D(nu_log,Er_list[index,:],D13[index,:,:],extrap=True)(grid_nu,grid_Er)
    x33=interpax.Interpolator2D(nu_log,Er_list[index,:],D33[index,:,:],extrap=True)(grid_nu,grid_Er)
    return x11,x13,x33

@jit
def interpolation_small_r(grid_x,grid_nu,grid_Er):
    xg=jnp.zeros(3)    
    xr2=jnp.power(grid_x,2)
    xr3=jnp.power(grid_x,3)
    r12 = jnp.power(r1,2)
    r22 = jnp.power(r2,2)
    r32 = jnp.power(r3,2)
    r13 = jnp.power(r1,3)
    r23 = jnp.power(r2,3)
    r33 = jnp.power(r3,3)
    x11_0,x13_0,x33_0=interpolator_nu_Er_general(0,grid_nu,grid_Er)
    x11_1,x13_1,x33_1=interpolator_nu_Er_general(1,grid_nu,grid_Er)
    x11_2,x13_2,x33_2=interpolator_nu_Er_general(2,grid_nu,grid_Er)
    ha1_11 = ((x11_2-x11_1)/(r33-r23)-(x11_2-x11_0)/(r33-r13))/((r32-r22)/(r33-r23)-(r32-r12)/(r33-r13))
    hb1_11 = ((x11_2-x11_1)/(r32-r22)-(x11_2-x11_0)/(r32-r12))/((r33-r23)/(r32-r22)-(r33-r13)/(r32-r12))
    hg1_11 = x11_0-r12*ha1_11-r13*hb1_11
    #13
    ha1_13 = ((x13_2-x13_1)/(r33-r23)-(x13_2-x13_0)/(r33-r13))/((r32-r22)/(r33-r23)-(r32-r12)/(r33-r13))
    hb1_13 = ((x13_2-x13_1)/(r32-r22)-(x13_2-x13_0)/(r32-r12))/((r33-r23)/(r32-r22)-(r33-r13)/(r32-r12))
    hg1_13 = x13_0-r12*ha1_13-r13*hb1_13
    #33
    ha1_33 = ((x33_2-x33_1)/(r33-r23)-(x33_2-x33_0)/(r33-r13))/((r32-r22)/(r33-r23)-(r32-r12)/(r33-r13))
    hb1_33 = ((x33_2-x33_1)/(r32-r22)-(x33_2-x33_0)/(r32-r12))/((r33-r23)/(r32-r22)-(r33-r13)/(r32-r12))
    hg1_33 = x33_0-r12*ha1_33-r13*hb1_33
    #Final output
    xg11  = hg1_11+xr2*ha1_11+xr3*hb1_11
    xg13  = hg1_13+xr2*ha1_13+xr3*hb1_13
    xg33  = hg1_33+xr2*ha1_33+xr3*hb1_33   
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg#xg11,xg13,xg33

@jit
def interpolation_large_r(grid_x,grid_nu,grid_Er):
    xg=jnp.zeros(3)    
    hr0 = (grid_x-rnm2)*(grid_x-rnm1)/((rnm3-rnm2)*(rnm3-rnm1))
    hr1 = (grid_x-rnm3)*(grid_x-rnm1)/((rnm2-rnm3)*(rnm2-rnm1))
    hr2 = (grid_x-rnm3)*(grid_x-rnm2)/((rnm1-rnm3)*(rnm1-rnm2))
    x11_m3,x13_m3,x33_m3=interpolator_nu_Er_general(-3,grid_nu,grid_Er)
    x11_m2,x13_m2,x33_m2=interpolator_nu_Er_general(-2,grid_nu,grid_Er)
    x11_m1,x13_m1,x33_m1=interpolator_nu_Er_general(-1,grid_nu,grid_Er)
    xg11  = hr0*x11_m3+hr1*x11_m2+hr2*x11_m1
    xg13  = hr0*x13_m3+hr1*x13_m2+hr2*x13_m1
    xg33  = hr0*x33_m3+hr1*x33_m2+hr2*x33_m1 
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg#xg11,xg13,xg33

@jit
def interpolation_mid_r(grid_x,grid_nu,grid_Er):
    xg=jnp.zeros(3)
    arr=grid_x-rho[1:-1]*a_b
    index = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf))+1
    idx0=index-2
    idx1=index-1
    idx2=index
    idx3=index+1
    ridx0=a_b*rho.at[idx0].get()
    ridx1=a_b*rho.at[idx1].get()
    ridx2=a_b*rho.at[idx2].get()
    ridx3=a_b*rho.at[idx3].get()
    #jax.debug.print("Pe {Pe} ", Pe=ind)
    hr0 = (grid_x-ridx1)*(grid_x-ridx2)*(grid_x-ridx3)/((ridx0-ridx1)*(ridx0-ridx2)*(ridx0-ridx3))
    hr1 = (grid_x-ridx0)*(grid_x-ridx2)*(grid_x-ridx3)/((ridx1-ridx0)*(ridx1-ridx2)*(ridx1-ridx3))
    hr2 = (grid_x-ridx0)*(grid_x-ridx1)*(grid_x-ridx3)/((ridx2-ridx0)*(ridx2-ridx1)*(ridx2-ridx3))
    hr3 = (grid_x-ridx0)*(grid_x-ridx1)*(grid_x-ridx2)/((ridx3-ridx0)*(ridx3-ridx1)*(ridx3-ridx2))
    x11_idx0,x13_idx0,x33_idx0=interpolator_nu_Er_general(idx0,grid_nu,grid_Er)
    x11_idx1,x13_idx1,x33_idx1=interpolator_nu_Er_general(idx1,grid_nu,grid_Er)
    x11_idx2,x13_idx2,x33_idx2=interpolator_nu_Er_general(idx2,grid_nu,grid_Er)
    x11_idx3,x13_idx3,x33_idx3=interpolator_nu_Er_general(idx3,grid_nu,grid_Er) 
    xg11  = hr0*x11_idx0+hr1*x11_idx1+hr2*x11_idx2+hr3*x11_idx3
    xg13  = hr0*x13_idx0+hr1*x13_idx1+hr2*x13_idx2+hr3*x13_idx3
    xg33  = hr0*x33_idx0+hr1*x33_idx1+hr2*x33_idx2+hr3*x33_idx3
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg#xg11,xg13,xg33


#def get_Dij(grid_x, grid_nu, grid_Er):
#  return jnp.select(condlist=[grid_x < a_b*rho[1], (grid_x >= a_b*rho[1] ) & (grid_x < a_b*rho[-2]), grid_x >= a_b*rho[-2] ],
#    choicelist=[interpolation_small_r(grid_x,grid_nu,grid_Er) ,interpolation_mid_r(grid_x,grid_nu,grid_Er)  , interpolation_large_r(grid_x,grid_nu,grid_Er) ],default=0)


def get_Dij_alt(grid_x, grid_nu, grid_Er):
    xg=jnp.zeros(3)
    #grid_nu_internal=jnp.log10(grid_nu)
    #grid_Er_internal=jnp.abs(grid_Er)
    #xg=jnp.select(condlist=[grid_x < r1_lim, (grid_x>=r1_lim) & (grid_x<rmn2_lim), grid_x >= rmn2_lim],
    #                  choicelist=[interpolation_small_r(grid_x,grid_nu_internal,grid_Er_internal) ,
    #                              interpolation_mid_r(grid_x,grid_nu_internal,grid_Er_internal) ,
    #                              interpolation_large_r(grid_x,grid_nu_internal,grid_Er_internal) ],default=0)
    xg11=interpax.Interpolator3D(rho*a_b,nu_log,Er_list[:],D11_log[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg13=interpax.Interpolator3D(rho*a_b,nu_log,Er_list[:],D13[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg33=interpax.Interpolator3D(rho*a_b,nu_log,Er_list[:],D33[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg
    #return interpax.Interpolator3D(rho*a_b,nu_log,Er_list,D11_log,extrap=True)(grid_x,grid_nu_internal,grid_Er_internal)

@jit
def get_Dij(grid_x, grid_nu, grid_Er):
    ##xg=jnp.zeros(3)
    grid_nu_internal=jnp.log10(jnp.maximum(1.e-12,grid_nu))
    grid_Er_internal=jnp.select(condlist=[grid_x<=low_limit_r,grid_x>low_limit_r], 
                              choicelist=[jnp.log10(Er_lower_limit),jnp.log10(jnp.maximum(Er_lower_limit,jnp.abs(grid_Er/grid_x)))],default=0)
    xg=jnp.select(condlist=[grid_x < r1_lim, (grid_x>=r1_lim) & (grid_x<rmn2_lim), grid_x >= rmn2_lim],
                      choicelist=[interpolation_small_r(grid_x,grid_nu_internal,grid_Er_internal) ,
                                  interpolation_mid_r(grid_x,grid_nu_internal,grid_Er_internal) ,
                                  interpolation_large_r(grid_x,grid_nu_internal,grid_Er_internal) ],default=0)
    return xg
    #return interpax.Interpolator3D(rho*a_b,nu_log,Er_list,D11_log,extrap=True)(grid_x,grid_nu_internal,grid_Er_internal)
