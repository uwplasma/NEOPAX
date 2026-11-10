import dataclasses

import jax.numpy as jnp
import jax
from jax import config
# to use higher precision
config.update("jax_enable_x64", True)
from jax import jit
import interpax
from interpax._coefs import A_BICUBIC
from ._native_interpolation import _cubic_apply, bicubic_channels, cubic1d


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MonoenergeticInterpolationStencil:
    """Compact selection record for one legacy ``get_Dij`` query.

    The legacy radial rule uses either three or four database surfaces.  Each
    selected bicubic surface query has support on at most four collisionality
    knots and four electric-field knots after transposing interpax's local C1
    slope construction.  Recording those discrete choices separately from
    the coefficient values is the first building block for a sparse database
    reverse boundary.
    """

    radial_indices: jax.Array
    radial_weights: jax.Array
    radial_weight_a_b_derivatives: jax.Array
    radial_active: jax.Array
    nu_upper_index: jax.Array
    nu_fraction: jax.Array
    er_upper_indices: jax.Array
    er_fractions: jax.Array
    grid_nu_internal: jax.Array
    grid_er_internal: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MonoenergeticSparseTableBar:
    """At most ``4 x 4 x 4`` coefficient updates for one scalar query."""

    radial_indices: jax.Array
    nu_indices: jax.Array
    er_indices: jax.Array
    values: jax.Array


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MonoenergeticSparseCoordinateBar:
    """Scalar scale bar and sparse ``Er_list`` updates for one query."""

    a_b: jax.Array
    radial_indices: jax.Array
    er_indices: jax.Array
    er_values: jax.Array



def interpolator_nu_low_Er0(index,grid_nu,database):
    #return D11_log[index,0,0]-(nu_log[0]-grid_nu)*(D11_log[index,1,0]-D11_log[index,0,0])/(nu_log[1]-nu_log[0])
    return database.D11_log[index,0,0]+(database.nu_log[0]-grid_nu)#*(D11_log[index,1,0]-D11_log[index,0,0])/(nu_log[1]-nu_log[0])


def interpolator_nu_mid_Er0(index,grid_nu,database):
    return cubic1d(database.nu_log, database.D11_log[index, :, 0], grid_nu)


def interpolator_nu_large_Er0(index,grid_nu,database):
    #return D11_log[index,-1,0]+(grid_nu-nu_log[-1])*(D11_log[index,-1,0]-D11_log[index,-2,0])/(nu_log[-1]-nu_log[-2])
    return database.D11_log[index,-1,0]+(grid_nu-database.nu_log[-1])#*(D11_log[index,-1,0]-D11_log[index,-2,0])/(nu_log[-1]-nu_log[-2])

def interpolator_nu_low_Er_finite(index,grid_nu,grid_Er,database):
    #Calculate Er_h
    Er_h=(database.nu_log[0]-grid_nu)/3.0+grid_Er  #This works with grid_Er as log(Er)
    return jnp.select(condlist=[Er_h<database.Er_list[index,-1],Er_h>=database.Er_list[index,-1]],
                      choicelist=[cubic1d(database.Er_list[index,:],database.D11_log[index,0,:],Er_h),database.D11_lower_limit],default=0)



def interpolator_nu_large_Er_finite(index,grid_nu,grid_Er,database):
    #This actually uses Tokamak + Er fit in NTSS!!! Have to update but for now I believe this is high collisionality so it will not affect much the comparison.
    # Thus mantaining general extrapolation 
    #TODO!! Create function for tokamak fit
    return bicubic_channels(database.nu_log, database.Er_list[index], (database.D11_log[index],), grid_nu, grid_Er)[0]



def interpolator_nu_mid_Er_finite_npoints3_low(index,grid_nu,grid_Er,database):
    #This is 3 points interpolation in nu
    #Calculate 3 points of nu for interpolatio
    nu0=database.nu_log.at[0].get()
    nu1=database.nu_log.at[1].get()
    nu2=database.nu_log.at[2].get()
    d11_nu=jnp.select(condlist=[grid_Er<=database.Er_list[index,0],
                     (grid_Er>database.Er_list[index,0])&(grid_Er<=database.Er_list[index,-1]),
                     grid_Er>database.Er_list[index,-1]],
                     choicelist=[jnp.array([database.D11_log.at[index,0,0].get(),database.D11_log.at[index,1,0].get(),database.D11_log.at[index,2,0].get()]),
                                  jnp.array([cubic1d(database.Er_list[index,:],database.D11_log[index,0,:],grid_Er),
                                   cubic1d(database.Er_list[index,:],database.D11_log[index,1,:],grid_Er),
                                   cubic1d(database.Er_list[index,:],database.D11_log[index,2,:],grid_Er)]),
                                  jnp.array([database.D11_lower_limit,database.D11_lower_limit,database.D11_lower_limit])],default=0)
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

def interpolator_nu_mid_Er_finite_npoints3_high(index,grid_nu,grid_Er,database):
    #This is 3 points interpolation in nu
    #Calculate 3 points of nu for interpolatio
    nu0=database.nu_log.at[-3].get()
    nu1=database.nu_log.at[-2].get()
    nu2=database.nu_log.at[-1].get()
    d11_nu=jnp.select(condlist=[grid_Er<=database.Er_list[index,0],
                     (grid_Er>database.Er_list[index,0])&(grid_Er<=database.Er_list[index,-1]),
                     grid_Er>database.Er_list[index,-1] ],
                     choicelist=[jnp.array([database.D11_log.at[index,-3,0].get(),database.D11_log.at[index,-2,0].get(),database.D11_log.at[index,-1,0].get()]),
                                  jnp.array([cubic1d(database.Er_list[index,:],database.D11_log[index,-3,:],grid_Er),
                                   cubic1d(database.Er_list[index,:],database.D11_log[index,-2,:],grid_Er),
                                   cubic1d(database.Er_list[index,:],database.D11_log[index,-1,:],grid_Er)]),
                                  jnp.array([database.D11_lower_limit,database.D11_lower_limit,database.D11_lower_limit])],default=0)
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


def interpolator_nu_mid_Er_finite_npoints4(index,grid_nu,grid_Er,database):
    #This is 4 points interpolation in nu
    #Calculate 4 points of nu for interpolation
    arr=grid_nu-database.nu_log[1:-1]
    index_nu = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf))+1
    idx0=index_nu-1
    idx1=index_nu-2
    idx2=index_nu
    idx3=index_nu+1
    nu0=database.nu_log.at[idx0].get()
    nu1=database.nu_log.at[idx1].get()
    nu2=database.nu_log.at[idx2].get()
    nu3=database.nu_log.at[idx3].get()
    d11_nu=jnp.select(condlist=[grid_Er<=database.Er_list[index,0],
                     (grid_Er>database.Er_list[index,0])&(grid_Er<=database.Er_list[index,-1]),
                     grid_Er>database.Er_list[index,-1] ],
                     choicelist=[jnp.array([database.D11_log.at[index,idx0,0].get(),database.D11_log.at[index,idx1,0].get(),database.D11_log.at[index,idx2,0].get(),database.D11_log.at[index,idx3,0].get()]),
                                  jnp.array([cubic1d(database.Er_list[index,:],database.D11_log[index,idx0,:],grid_Er),
                                   cubic1d(database.Er_list[index,:],database.D11_log[index,idx1,:],grid_Er),
                                   cubic1d(database.Er_list[index,:],database.D11_log[index,idx2,:],grid_Er),
                                   cubic1d(database.Er_list[index,:],database.D11_log[index,idx3,:],grid_Er)]),
                                  jnp.array([database.D11_lower_limit,database.D11_lower_limit,database.D11_lower_limit,database.D11_lower_limit])],default=0)
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
        xnun  = (grid_nu-nu1[idx1])/dxnu21
        ha1   = 3.*(d11_2-d11_1)-(2.*dg11_new[0]+dg11_new[1])*dxnu21
        hb1   =-2.*(d11_2-d11_1)+(dg11_new[0]+dg11_new[1])*dxnu21
        xg11 = d11_1+xnun*(dg11_new[0]*dxnu21+xnun*(ha1+xnun*hb1))
        return xg11
    return jnp.select(condlist=[d11_nu[2]<=database.D11_lower_limit,d11_nu[2]>database.D11_lower_limit],
                      choicelist=[database.D11_lower_limit,nu_npoints4_interpolation(nu0,nu1,nu2,nu3,grid_nu,d11_0,d11_1,d11_2,d11_3)],default=0)





def interpolator_nu_Er_general_NTSS(index,grid_nu,grid_Er,database):
    x11=jnp.select(condlist=[(grid_Er<=database.Er_lower_limit_log) & (grid_nu < database.nu_log[0]),   #nu_low Er_low
                                (grid_Er<=database.Er_lower_limit_log) & ((grid_nu >= database.nu_log[0]) & (grid_nu <= database.nu_log[-1])), #nu_mid, Er_low
                                (grid_Er<=database.Er_lower_limit_log) & (grid_nu > database.nu_log[-1]), #nu_high, Er_low
                                (grid_Er>database.Er_lower_limit_log) & (grid_nu < database.nu_log[0]),   #nu_low, Er_finite
                                (grid_Er>database.Er_lower_limit_log) & ((grid_nu >= database.nu_log[0]) & (grid_nu <= database.nu_log[1])),  #between nu[0] and nu[1], with Er finite, 3 points
                                (grid_Er>database.Er_lower_limit_log) & ((grid_nu >= database.nu_log[-2]) & (grid_nu <= database.nu_log[-1])),  #between nu[-2] and nu[-1], with Er finite, 3 points
                                (grid_Er>database.Er_lower_limit_log) & ((grid_nu > database.nu_log[1]) & (grid_nu < database.nu_log[-2])),  #between nu[1] and nu[2] (mid nu), with Er finite, 4 points
                                ###(grid_Er>Er_lower_limit) & ((grid_nu > nu_log[0]) & (grid_nu < nu_log[-1])),  #between nu[1] and nu[2] (mid nu), with Er finite, 4 points                                
                                (grid_Er>database.Er_lower_limit_log) & (grid_nu > database.nu_log[-1]) #nu_large, Er_finite
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
    x13,x33=bicubic_channels(database.nu_log, database.Er_list[index], (database.D13[index], database.D33[index]), grid_nu, grid_Er)
    return x11,x13,x33                                





@jit
#####Interpolators, which should go to the _interpolators.py
def interpolator_nu_Er_general(index,grid_nu,grid_Er,database):
    return bicubic_channels(
        database.nu_log, database.Er_list[index],
        (database.D11_log[index], database.D13[index], database.D33[index]),
        grid_nu, grid_Er,
    )

def _interpolated_surfaces(indices, grid_nu, grid_Er, database):
    """Evaluate one local radial stencil without separate per-surface graphs."""
    values = jax.vmap(interpolator_nu_Er_general, in_axes=(0, None, None, None))(
        indices, grid_nu, grid_Er, database
    )
    return jnp.stack(values, axis=-1)


@jit
def interpolation_small_r(grid_x, grid_nu, grid_Er, database):
    v0, v1, v2 = _interpolated_surfaces(jnp.arange(3, dtype=jnp.int32), grid_nu, grid_Er, database)
    r12, r22, r32 = database.r1**2, database.r2**2, database.r3**2
    r13, r23, r33 = database.r1**3, database.r2**3, database.r3**3
    a = ((v2-v1)/(r33-r23) - (v2-v0)/(r33-r13)) / ((r32-r22)/(r33-r23) - (r32-r12)/(r33-r13))
    b = ((v2-v1)/(r32-r22) - (v2-v0)/(r32-r12)) / ((r33-r23)/(r32-r22) - (r33-r13)/(r32-r12))
    return (v0 - r12*a - r13*b) + grid_x**2*a + grid_x**3*b

@jit
def interpolation_large_r(grid_x,grid_nu,grid_Er,database):
    hr0 = (grid_x-database.rnm2)*(grid_x-database.rnm1)/((database.rnm3-database.rnm2)*(database.rnm3-database.rnm1))
    hr1 = (grid_x-database.rnm3)*(grid_x-database.rnm1)/((database.rnm2-database.rnm3)*(database.rnm2-database.rnm1))
    hr2 = (grid_x-database.rnm3)*(grid_x-database.rnm2)/((database.rnm1-database.rnm3)*(database.rnm1-database.rnm2))
    indices = jnp.arange(database.rho.size - 3, database.rho.size, dtype=jnp.int32)
    v0, v1, v2 = _interpolated_surfaces(indices, grid_nu, grid_Er, database)
    return hr0*v0 + hr1*v1 + hr2*v2

@jit
def interpolation_mid_r(grid_x,grid_nu,grid_Er,database):
    arr=grid_x-database.rho[1:-1]*database.a_b
    index = jnp.argmax(jnp.where(arr <= 0, arr, -jnp.inf))+1
    indices = index + jnp.asarray((-2, -1, 0, 1), dtype=jnp.int32)
    ridx0, ridx1, ridx2, ridx3 = database.a_b * database.rho[indices]
    hr0 = (grid_x-ridx1)*(grid_x-ridx2)*(grid_x-ridx3)/((ridx0-ridx1)*(ridx0-ridx2)*(ridx0-ridx3))
    hr1 = (grid_x-ridx0)*(grid_x-ridx2)*(grid_x-ridx3)/((ridx1-ridx0)*(ridx1-ridx2)*(ridx1-ridx3))
    hr2 = (grid_x-ridx0)*(grid_x-ridx1)*(grid_x-ridx3)/((ridx2-ridx0)*(ridx2-ridx1)*(ridx2-ridx3))
    hr3 = (grid_x-ridx0)*(grid_x-ridx1)*(grid_x-ridx2)/((ridx3-ridx0)*(ridx3-ridx1)*(ridx3-ridx2))
    v0, v1, v2, v3 = _interpolated_surfaces(indices, grid_nu, grid_Er, database)
    return hr0*v0 + hr1*v1 + hr2*v2 + hr3*v3


#def get_Dij(grid_x, grid_nu, grid_Er):
#  return jnp.select(condlist=[grid_x < a_b*rho[1], (grid_x >= a_b*rho[1] ) & (grid_x < a_b*rho[-2]), grid_x >= a_b*rho[-2] ],
#    choicelist=[interpolation_small_r(grid_x,grid_nu,grid_Er) ,interpolation_mid_r(grid_x,grid_nu,grid_Er)  , interpolation_large_r(grid_x,grid_nu,grid_Er) ],default=0)


def get_Dij_alt(grid_x, grid_nu, grid_Er,database):
    xg=jnp.zeros(3)
    #grid_nu_internal=jnp.log10(grid_nu)
    #grid_Er_internal=jnp.abs(grid_Er)
    #xg=jnp.select(condlist=[grid_x < r1_lim, (grid_x>=r1_lim) & (grid_x<rmn2_lim), grid_x >= rmn2_lim],
    #                  choicelist=[interpolation_small_r(grid_x,grid_nu_internal,grid_Er_internal) ,
    #                              interpolation_mid_r(grid_x,grid_nu_internal,grid_Er_internal) ,
    #                              interpolation_large_r(grid_x,grid_nu_internal,grid_Er_internal) ],default=0)
    xg11=interpax.Interpolator3D(database.rho*database.a_b,database.nu_log,database.Er_list[:],database.D11_log[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg13=interpax.Interpolator3D(database.rho*database.a_b,database.nu_log,database.Er_list[:],database.D13[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg33=interpax.Interpolator3D(database.rho*database.a_b,database.nu_log,database.Er_list[:],database.D33[:,:,:],extrap=True)(grid_x,jnp.log10(grid_nu),jnp.abs(grid_Er))
    xg=xg.at[0].set(xg11)
    xg=xg.at[1].set(xg13)
    xg=xg.at[2].set(xg33)
    return xg
    #return interpax.Interpolator3D(rho*a_b,nu_log,Er_list,D11_log,extrap=True)(grid_x,grid_nu_internal,grid_Er_internal)


@jit
def get_Dij_3d(grid_x, grid_nu, grid_Er, database):
    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    # A true 3D tensor grid cannot use database.Er_list directly because that
    # axis is radius-dependent (it stores log10(|Er|/r)). Reconstruct a common
    # log10(|Er|) axis from the first radius row and use that for the 3D query.
    grid_Er_internal = jnp.log10(jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er)))
    r_grid = database.rho * database.a_b
    r_ref = jnp.max(r_grid)
    er_raw_log_grid = database.Er_list[0, :] + jnp.log10(jnp.maximum(r_ref, 1.0e-30))
    xg11 = interpax.Interpolator3D(
        r_grid,
        database.nu_log,
        er_raw_log_grid,
        database.D11_log[:, :, :],
        extrap=True,
    )(grid_x, grid_nu_internal, grid_Er_internal)
    xg13 = interpax.Interpolator3D(
        r_grid,
        database.nu_log,
        er_raw_log_grid,
        database.D13[:, :, :],
        extrap=True,
    )(grid_x, grid_nu_internal, grid_Er_internal)
    xg33 = interpax.Interpolator3D(
        r_grid,
        database.nu_log,
        er_raw_log_grid,
        database.D33[:, :, :],
        extrap=True,
    )(grid_x, grid_nu_internal, grid_Er_internal)
    return jnp.asarray([xg11, xg13, xg33])

@jit
def get_Dij(grid_x, grid_nu, grid_Er,database):
    grid_nu_internal=jnp.log10(jnp.maximum(1.e-12,grid_nu))
    safe_grid_x = jnp.where(grid_x == 0., 1., grid_x)
    grid_Er_internal=jnp.select(condlist=[grid_x<=database.low_limit_r,grid_x>database.low_limit_r],
                              choicelist=[jnp.log10(database.Er_lower_limit),jnp.log10(jnp.maximum(database.Er_lower_limit,jnp.abs(grid_Er/safe_grid_x)))],default=0)
    branch = jnp.where(grid_x < database.r1_lim, 0, jnp.where(grid_x < database.rmn2_lim, 1, 2))
    # A scalar radial query needs one closure, not all ten surface queries.
    # The discrete branch remains frozen by AD, as in the previous mask rule.
    return jax.lax.switch(
        branch, (interpolation_small_r, interpolation_mid_r, interpolation_large_r),
        grid_x, grid_nu_internal, grid_Er_internal, database,
    )


@jax.jit
def monoenergetic_interpolation_stencil(grid_x, grid_nu, grid_Er, database):
    """Record the active radial and bicubic intervals of legacy ``get_Dij``.

    This has the same piecewise radial contract as ``get_Dij`` but evaluates
    no coefficient table.  Three-point branches are zero-padded to four
    entries so batches of centre and face queries have one static shape.
    """

    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    safe_grid_x = jnp.where(
        jnp.abs(grid_x) > 0.0,
        grid_x,
        jnp.asarray(1.0, dtype=jnp.asarray(grid_x).dtype),
    )
    grid_er_internal = jnp.where(
        grid_x <= database.low_limit_r,
        jnp.log10(database.Er_lower_limit),
        jnp.log10(
            jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er / safe_grid_x))
        ),
    )

    # Small-radius cubic-in-r closure, written as three linear weights.  This
    # is algebraically identical to ``interpolation_small_r`` and to the
    # established dense table transpose below.
    r12, r22, r32 = database.r1**2, database.r2**2, database.r3**2
    r13, r23, r33 = database.r1**3, database.r2**3, database.r3**3
    xr2, xr3 = grid_x**2, grid_x**3
    denom_a = (r32-r22)/(r33-r23) - (r32-r12)/(r33-r13)
    denom_b = (r33-r23)/(r32-r22) - (r33-r13)/(r32-r12)
    small_a = jnp.asarray((
        1.0/(r33-r13)/denom_a,
        -1.0/(r33-r23)/denom_a,
        (1.0/(r33-r23)-1.0/(r33-r13))/denom_a,
    ))
    small_b = jnp.asarray((
        1.0/(r32-r12)/denom_b,
        -1.0/(r32-r22)/denom_b,
        (1.0/(r32-r22)-1.0/(r32-r12))/denom_b,
    ))
    small_weights = jnp.concatenate((
        jnp.asarray((1.0, 0.0, 0.0))
        + (xr2-r12)*small_a
        + (xr3-r13)*small_b,
        jnp.zeros((1,), dtype=jnp.asarray(grid_x).dtype),
    ))
    small_indices = jnp.asarray((0, 1, 2, 2), dtype=jnp.int32)

    # Interior four-point Lagrange closure.  Preserve the legacy interval
    # selection exactly; its integer choice is intentionally not
    # differentiated.
    radial_offset = grid_x - database.rho[1:-1] * database.a_b
    mid_index = (
        jnp.argmax(jnp.where(radial_offset <= 0.0, radial_offset, -jnp.inf)) + 1
    )
    mid_indices = mid_index + jnp.asarray((-2, -1, 0, 1), dtype=jnp.int32)
    mid_radii = database.a_b * database.rho[mid_indices]
    mid_weights = jnp.asarray((
        (grid_x-mid_radii[1])*(grid_x-mid_radii[2])*(grid_x-mid_radii[3])
        / ((mid_radii[0]-mid_radii[1])*(mid_radii[0]-mid_radii[2])*(mid_radii[0]-mid_radii[3])),
        (grid_x-mid_radii[0])*(grid_x-mid_radii[2])*(grid_x-mid_radii[3])
        / ((mid_radii[1]-mid_radii[0])*(mid_radii[1]-mid_radii[2])*(mid_radii[1]-mid_radii[3])),
        (grid_x-mid_radii[0])*(grid_x-mid_radii[1])*(grid_x-mid_radii[3])
        / ((mid_radii[2]-mid_radii[0])*(mid_radii[2]-mid_radii[1])*(mid_radii[2]-mid_radii[3])),
        (grid_x-mid_radii[0])*(grid_x-mid_radii[1])*(grid_x-mid_radii[2])
        / ((mid_radii[3]-mid_radii[0])*(mid_radii[3]-mid_radii[1])*(mid_radii[3]-mid_radii[2])),
    ))

    # Edge three-point Lagrange closure.
    n_radius = database.rho.shape[0]
    large_indices = jnp.asarray(
        (n_radius - 3, n_radius - 2, n_radius - 1, n_radius - 1),
        dtype=jnp.int32,
    )
    large_radii = jnp.asarray((database.rnm3, database.rnm2, database.rnm1))
    large_weights = jnp.concatenate((jnp.asarray((
        (grid_x-large_radii[1])*(grid_x-large_radii[2])
        / ((large_radii[0]-large_radii[1])*(large_radii[0]-large_radii[2])),
        (grid_x-large_radii[0])*(grid_x-large_radii[2])
        / ((large_radii[1]-large_radii[0])*(large_radii[1]-large_radii[2])),
        (grid_x-large_radii[0])*(grid_x-large_radii[1])
        / ((large_radii[2]-large_radii[0])*(large_radii[2]-large_radii[1])),
    )), jnp.zeros((1,), dtype=jnp.asarray(grid_x).dtype)))

    radial_branch = jnp.where(
        grid_x < database.r1_lim,
        0,
        jnp.where(grid_x < database.rmn2_lim, 1, 2),
    )
    radial_indices = jnp.where(
        radial_branch == 0,
        small_indices,
        jnp.where(radial_branch == 1, mid_indices, large_indices),
    )
    radial_weights = jnp.where(
        radial_branch == 0,
        small_weights,
        jnp.where(radial_branch == 1, mid_weights, large_weights),
    )

    def _selected_radial_weights_for_scale(a_b):
        small_radii = a_b * database.rho[:3]
        sr12, sr22, sr32 = small_radii**2
        sr13, sr23, sr33 = small_radii**3
        scale_denom_a = (
            (sr32-sr22)/(sr33-sr23) - (sr32-sr12)/(sr33-sr13)
        )
        scale_denom_b = (
            (sr33-sr23)/(sr32-sr22) - (sr33-sr13)/(sr32-sr12)
        )
        scale_small_a = jnp.asarray((
            1.0/(sr33-sr13)/scale_denom_a,
            -1.0/(sr33-sr23)/scale_denom_a,
            (1.0/(sr33-sr23)-1.0/(sr33-sr13))/scale_denom_a,
        ))
        scale_small_b = jnp.asarray((
            1.0/(sr32-sr12)/scale_denom_b,
            -1.0/(sr32-sr22)/scale_denom_b,
            (1.0/(sr32-sr22)-1.0/(sr32-sr12))/scale_denom_b,
        ))
        scale_small_weights = jnp.concatenate((
            jnp.asarray((1.0, 0.0, 0.0))
            + (xr2-sr12)*scale_small_a
            + (xr3-sr13)*scale_small_b,
            jnp.zeros((1,), dtype=jnp.asarray(grid_x).dtype),
        ))

        scale_mid_radii = a_b * database.rho[mid_indices]
        scale_mid_weights = jnp.asarray((
            (grid_x-scale_mid_radii[1])*(grid_x-scale_mid_radii[2])*(grid_x-scale_mid_radii[3])
            / ((scale_mid_radii[0]-scale_mid_radii[1])*(scale_mid_radii[0]-scale_mid_radii[2])*(scale_mid_radii[0]-scale_mid_radii[3])),
            (grid_x-scale_mid_radii[0])*(grid_x-scale_mid_radii[2])*(grid_x-scale_mid_radii[3])
            / ((scale_mid_radii[1]-scale_mid_radii[0])*(scale_mid_radii[1]-scale_mid_radii[2])*(scale_mid_radii[1]-scale_mid_radii[3])),
            (grid_x-scale_mid_radii[0])*(grid_x-scale_mid_radii[1])*(grid_x-scale_mid_radii[3])
            / ((scale_mid_radii[2]-scale_mid_radii[0])*(scale_mid_radii[2]-scale_mid_radii[1])*(scale_mid_radii[2]-scale_mid_radii[3])),
            (grid_x-scale_mid_radii[0])*(grid_x-scale_mid_radii[1])*(grid_x-scale_mid_radii[2])
            / ((scale_mid_radii[3]-scale_mid_radii[0])*(scale_mid_radii[3]-scale_mid_radii[1])*(scale_mid_radii[3]-scale_mid_radii[2])),
        ))

        scale_large_radii = a_b * database.rho[-3:]
        scale_large_weights = jnp.concatenate((jnp.asarray((
            (grid_x-scale_large_radii[1])*(grid_x-scale_large_radii[2])
            / ((scale_large_radii[0]-scale_large_radii[1])*(scale_large_radii[0]-scale_large_radii[2])),
            (grid_x-scale_large_radii[0])*(grid_x-scale_large_radii[2])
            / ((scale_large_radii[1]-scale_large_radii[0])*(scale_large_radii[1]-scale_large_radii[2])),
            (grid_x-scale_large_radii[0])*(grid_x-scale_large_radii[1])
            / ((scale_large_radii[2]-scale_large_radii[0])*(scale_large_radii[2]-scale_large_radii[1])),
        )), jnp.zeros((1,), dtype=jnp.asarray(grid_x).dtype)))
        return jnp.where(
            radial_branch == 0,
            scale_small_weights,
            jnp.where(
                radial_branch == 1,
                scale_mid_weights,
                scale_large_weights,
            ),
        )

    _, radial_weight_a_b_derivatives = jax.jvp(
        _selected_radial_weights_for_scale,
        (jnp.asarray(database.a_b),),
        (jnp.ones_like(jnp.asarray(database.a_b)),),
    )
    radial_active = jnp.arange(4, dtype=jnp.int32) < jnp.where(
        radial_branch == 1, 4, 3
    )

    nu_upper_index = jnp.clip(
        jnp.searchsorted(database.nu_log, grid_nu_internal, side="right"),
        1,
        database.nu_log.shape[0] - 1,
    )
    nu_dx = (
        database.nu_log[nu_upper_index]
        - database.nu_log[nu_upper_index - 1]
    )
    nu_fraction = (grid_nu_internal - database.nu_log[nu_upper_index - 1]) * jnp.where(
        nu_dx == 0.0, 0.0, 1.0 / nu_dx
    )

    def _er_interval(surface_index):
        er_grid = jax.lax.dynamic_index_in_dim(
            database.Er_list, surface_index, axis=0, keepdims=False
        )
        upper = jnp.clip(
            jnp.searchsorted(er_grid, grid_er_internal, side="right"),
            1,
            er_grid.shape[0] - 1,
        )
        er_dx = er_grid[upper] - er_grid[upper - 1]
        fraction = (grid_er_internal - er_grid[upper - 1]) * jnp.where(
            er_dx == 0.0, 0.0, 1.0 / er_dx
        )
        return upper, fraction

    er_upper_indices, er_fractions = jax.vmap(_er_interval)(radial_indices)
    return MonoenergeticInterpolationStencil(
        radial_indices=radial_indices,
        radial_weights=radial_weights,
        radial_weight_a_b_derivatives=radial_weight_a_b_derivatives,
        radial_active=radial_active,
        nu_upper_index=nu_upper_index,
        nu_fraction=nu_fraction,
        er_upper_indices=er_upper_indices,
        er_fractions=er_fractions,
        grid_nu_internal=grid_nu_internal,
        grid_er_internal=grid_er_internal,
    )


@jax.jit
def evaluate_monoenergetic_interpolation_stencil(stencil, database):
    """Evaluate a legacy monoenergetic query from its selected stencil."""

    def _surface_value(surface_index):
        return jnp.asarray(
            interpolator_nu_Er_general(
                surface_index,
                stencil.grid_nu_internal,
                stencil.grid_er_internal,
                database,
            )
        )

    surface_values = jax.vmap(_surface_value)(stencil.radial_indices)
    weights = jnp.where(stencil.radial_active, stencil.radial_weights, 0.0)
    return jnp.einsum("r,rc->c", weights, surface_values)


def _cubic1_transpose(table_bar, knots, axis):
    """Transpose interpax's local C1 ``approx_df(..., "cubic")`` rule.

    The legacy ``Monoenergetic`` interpolation uses this exact derivative
    construction before its bicubic Hermite evaluation.  Keeping its
    transpose here makes the database-table reverse explicit: no VJP of an
    ``Interpolator2D`` (and consequently no interpolation tape) is built in
    the transport reverse sweep.
    """

    moved = jnp.moveaxis(table_bar, axis, 0)
    slope_bar = jnp.zeros_like(moved[:-1])
    slope_bar = slope_bar.at[0].add(moved[0])
    slope_bar = slope_bar.at[-1].add(moved[-1])
    interior = moved[1:-1]
    slope_bar = slope_bar.at[:-1].add(0.5 * interior)
    slope_bar = slope_bar.at[1:].add(0.5 * interior)
    inv_dx = jnp.where(jnp.diff(knots) == 0.0, 0.0, 1.0 / jnp.diff(knots))
    inv_dx = inv_dx.reshape((inv_dx.shape[0],) + (1,) * (slope_bar.ndim - 1))
    result = jnp.zeros_like(moved)
    result = result.at[:-1].add(-inv_dx * slope_bar)
    result = result.at[1:].add(inv_dx * slope_bar)
    return jnp.moveaxis(result, 0, axis)


def _monoenergetic_slice_table_bar(nu_grid, er_grid, grid_nu, grid_er, local_bar, table):
    """Exact table transpose of one legacy interpax bicubic surface query."""

    i = jnp.clip(jnp.searchsorted(nu_grid, grid_nu, side="right"), 1, nu_grid.shape[0] - 1)
    j = jnp.clip(jnp.searchsorted(er_grid, grid_er, side="right"), 1, er_grid.shape[0] - 1)
    dx = nu_grid[i] - nu_grid[i - 1]
    dy = er_grid[j] - er_grid[j - 1]
    tx = (grid_nu - nu_grid[i - 1]) * jnp.where(dx == 0.0, 0.0, 1.0 / dx)
    ty = (grid_er - er_grid[j - 1]) * jnp.where(dy == 0.0, 0.0, 1.0 / dy)

    # ``interpax.interp2d`` forms ``coef = A_BICUBIC @ F`` and then evaluates
    # ``sum_ij coef_ij [1,t,t^2,t^3]_i [1,u,u^2,u^3]_j``.  Materialize only
    # this 16-entry local linear map and scatter its transpose below.
    powers_x = jnp.asarray((1.0, tx, tx * tx, tx * tx * tx), dtype=table.dtype)
    powers_y = jnp.asarray((1.0, ty, ty * ty, ty * ty * ty), dtype=table.dtype)
    bicubic = jnp.asarray(A_BICUBIC, dtype=table.dtype)
    coefficient_basis = bicubic @ jnp.eye(16, dtype=table.dtype)
    coefficient_basis = jnp.reshape(coefficient_basis, (4, 4, 16), order="F")
    f_weights = local_bar * jnp.einsum(
        "ij,ijm->m", jnp.outer(powers_x, powers_y), coefficient_basis
    )

    def _scatter_corners(result, corner_bar):
        result = result.at[i - 1, j - 1].add(corner_bar[0])
        result = result.at[i, j - 1].add(corner_bar[1])
        result = result.at[i - 1, j].add(corner_bar[2])
        return result.at[i, j].add(corner_bar[3])

    direct_bar = _scatter_corners(jnp.zeros_like(table), f_weights[:4])
    fx_bar = _scatter_corners(jnp.zeros_like(table), dx * f_weights[4:8])
    fy_bar = _scatter_corners(jnp.zeros_like(table), dy * f_weights[8:12])
    fxy_bar = _scatter_corners(jnp.zeros_like(table), dx * dy * f_weights[12:])
    return (
        direct_bar
        + _cubic1_transpose(fx_bar, nu_grid, axis=0)
        + _cubic1_transpose(fy_bar, er_grid, axis=1)
        + _cubic1_transpose(
            _cubic1_transpose(fxy_bar, er_grid, axis=1), nu_grid, axis=0
        )
    )


def _cubic1_pair_transpose_weights(knots, upper_index):
    """Transpose C1 slopes at one interval's two endpoints into four slots."""

    inverse_spacing = jnp.where(
        jnp.diff(knots) == 0.0, 0.0, 1.0 / jnp.diff(knots)
    )

    def _lower_boundary():
        inverse_dx = inverse_spacing[0]
        return jnp.zeros((4,), dtype=knots.dtype).at[1].set(
            -inverse_dx
        ).at[2].set(inverse_dx)

    def _lower_interior():
        inverse_left = inverse_spacing[upper_index - 2]
        inverse_right = inverse_spacing[upper_index - 1]
        return jnp.asarray((
            -0.5 * inverse_left,
            0.5 * inverse_left - 0.5 * inverse_right,
            0.5 * inverse_right,
            0.0,
        ))

    def _upper_boundary():
        inverse_dx = inverse_spacing[upper_index - 1]
        return jnp.zeros((4,), dtype=knots.dtype).at[1].set(
            -inverse_dx
        ).at[2].set(inverse_dx)

    def _upper_interior():
        inverse_left = inverse_spacing[upper_index - 1]
        inverse_right = inverse_spacing[upper_index]
        return jnp.asarray((
            0.0,
            -0.5 * inverse_left,
            0.5 * inverse_left - 0.5 * inverse_right,
            0.5 * inverse_right,
        ))

    lower_weights = jax.lax.cond(
        upper_index == 1, _lower_boundary, _lower_interior
    )
    upper_weights = jax.lax.cond(
        upper_index == knots.shape[0] - 1,
        _upper_boundary,
        _upper_interior,
    )
    return jnp.stack((lower_weights, upper_weights))


def _monoenergetic_slice_sparse_table_bar(
    nu_grid,
    er_grid,
    nu_upper_index,
    er_upper_index,
    nu_fraction,
    er_fraction,
    local_bar,
    table,
):
    """Exact 4-by-4 table support of one bicubic surface transpose."""

    dx = nu_grid[nu_upper_index] - nu_grid[nu_upper_index - 1]
    dy = er_grid[er_upper_index] - er_grid[er_upper_index - 1]
    tx = nu_fraction
    ty = er_fraction
    powers_x = jnp.asarray((1.0, tx, tx * tx, tx * tx * tx), dtype=table.dtype)
    powers_y = jnp.asarray((1.0, ty, ty * ty, ty * ty * ty), dtype=table.dtype)
    bicubic = jnp.asarray(A_BICUBIC, dtype=table.dtype)
    coefficient_basis = bicubic @ jnp.eye(16, dtype=table.dtype)
    coefficient_basis = jnp.reshape(
        coefficient_basis, (4, 4, 16), order="F"
    )
    f_weights = local_bar * jnp.einsum(
        "ij,ijm->m", jnp.outer(powers_x, powers_y), coefficient_basis
    )

    def _corners(values):
        return jnp.asarray(((values[0], values[2]), (values[1], values[3])))

    direct_bar = _corners(f_weights[:4])
    fx_bar = dx * _corners(f_weights[4:8])
    fy_bar = dy * _corners(f_weights[8:12])
    fxy_bar = dx * dy * _corners(f_weights[12:])
    nu_slope_transpose = _cubic1_pair_transpose_weights(
        nu_grid, nu_upper_index
    )
    er_slope_transpose = _cubic1_pair_transpose_weights(
        er_grid, er_upper_index
    )

    result = jnp.zeros((4, 4), dtype=table.dtype)
    result = result.at[1:3, 1:3].add(direct_bar)
    result = result.at[:, 1:3].add(nu_slope_transpose.T @ fx_bar)
    result = result.at[1:3, :].add(fy_bar @ er_slope_transpose)
    return result + nu_slope_transpose.T @ fxy_bar @ er_slope_transpose


@jax.jit
def monoenergetic_interpolation_sparse_table_bar(
    stencil, local_bar, table, database
):
    """Return the nonzero local table transpose of one stencil query.

    The returned values have shape ``(4, 4, 4)``: radial slot,
    collisionality support, and electric-field support.  Invalid boundary
    padding and the fourth slot of a three-surface radial branch are exactly
    zero.  In particular, this function does not return a full database-sized
    cotangent for every scalar query.
    """

    support_offsets = jnp.asarray((-2, -1, 0, 1), dtype=jnp.int32)
    nu_raw_indices = stencil.nu_upper_index + support_offsets
    nu_active = (nu_raw_indices >= 0) & (nu_raw_indices < table.shape[1])
    nu_indices = jnp.clip(nu_raw_indices, 0, table.shape[1] - 1)

    def _surface_sparse_bar(slot):
        surface_index = stencil.radial_indices[slot]
        er_grid = jax.lax.dynamic_index_in_dim(
            database.Er_list, surface_index, axis=0, keepdims=False
        )
        surface_table = jax.lax.dynamic_index_in_dim(
            table, surface_index, axis=0, keepdims=False
        )
        values = _monoenergetic_slice_sparse_table_bar(
            database.nu_log,
            er_grid,
            stencil.nu_upper_index,
            stencil.er_upper_indices[slot],
            stencil.nu_fraction,
            stencil.er_fractions[slot],
            stencil.radial_weights[slot] * local_bar,
            surface_table,
        )
        er_raw_indices = stencil.er_upper_indices[slot] + support_offsets
        er_active = (er_raw_indices >= 0) & (
            er_raw_indices < surface_table.shape[1]
        )
        er_indices = jnp.clip(er_raw_indices, 0, surface_table.shape[1] - 1)
        active = (
            stencil.radial_active[slot]
            & nu_active[:, None]
            & er_active[None, :]
        )
        return er_indices, jnp.where(active, values, 0.0)

    er_indices, values = jax.vmap(_surface_sparse_bar)(
        jnp.arange(4, dtype=jnp.int32)
    )
    return MonoenergeticSparseTableBar(
        radial_indices=stencil.radial_indices,
        nu_indices=nu_indices,
        er_indices=er_indices,
        values=values,
    )


@jax.jit
def monoenergetic_interpolation_sparse_coordinate_bar(stencil, local_bar, database):
    """Transpose a recorded query using only its local cubic knot support."""
    offsets = jnp.arange(-2, 2, dtype=jnp.int32)
    nu_upper = stencil.nu_upper_index
    ix = jnp.clip(nu_upper + offsets, 0, database.nu_log.size - 1)
    spacing = jnp.diff(database.nu_log[ix])
    inverse = jnp.where(spacing == 0., 0., 1. / jnp.where(spacing == 0., 1., spacing))
    x_geometry = (inverse, spacing[1], stencil.nu_fraction,
                  nu_upper == 1, nu_upper == database.nu_log.size - 1)

    def surface(slot):
        radius = stencil.radial_indices[slot]
        upper = stencil.er_upper_indices[slot]
        iy = jnp.clip(upper + offsets, 0, database.Er_list.shape[1] - 1)
        knots = database.Er_list[radius, iy]
        tables = jnp.stack(tuple(table[radius, ix[:, None], iy[None, :]]
                                 for table in (database.D11_log, database.D13, database.D33)), axis=-1)
        values = _cubic_apply(tables, x_geometry)

        def coefficients(ys):
            spacing = jnp.diff(ys)
            inverse = jnp.where(spacing == 0., 0., 1. / jnp.where(spacing == 0., 1., spacing))
            width = spacing[1]
            t = (stencil.grid_er_internal - ys[1]) * inverse[1]
            geometry = inverse, width, t, upper == 1, upper == database.Er_list.shape[1] - 1
            value = _cubic_apply(values, geometry)
            return value, value

        jacobian, value = jax.jacfwd(coefficients, has_aux=True)(knots)
        raw = jnp.vdot(local_bar, value)
        bars = jnp.einsum("c,ck->k", local_bar, jacobian) * stencil.radial_weights[slot]
        active = stencil.radial_active[slot]
        return iy, jnp.where(active, bars, 0.), jnp.where(active, raw, 0.)

    indices, bars, raw = jax.vmap(surface)(jnp.arange(4, dtype=jnp.int32))
    return MonoenergeticSparseCoordinateBar(
        a_b=jnp.vdot(stencil.radial_weight_a_b_derivatives, raw),
        radial_indices=stencil.radial_indices, er_indices=indices, er_values=bars)


@jax.jit
def materialize_monoenergetic_sparse_coordinate_bar(coordinate_bar, database):
    """Scatter a compact coordinate cotangent for focused parity checks."""

    radial_indices = jnp.broadcast_to(
        coordinate_bar.radial_indices[:, None], coordinate_bar.er_values.shape
    )
    return (
        coordinate_bar.a_b,
        jnp.zeros_like(database.Er_list).at[
            radial_indices, coordinate_bar.er_indices
        ].add(coordinate_bar.er_values),
    )


@jax.jit
def materialize_monoenergetic_sparse_table_bar(sparse_bar, table):
    """Scatter one compact table cotangent into a dense table for parity."""

    radial_indices = jnp.broadcast_to(
        sparse_bar.radial_indices[:, None, None], sparse_bar.values.shape
    )
    nu_indices = jnp.broadcast_to(
        sparse_bar.nu_indices[None, :, None], sparse_bar.values.shape
    )
    er_indices = jnp.broadcast_to(
        sparse_bar.er_indices[:, None, :], sparse_bar.values.shape
    )
    return jnp.zeros_like(table).at[
        radial_indices, nu_indices, er_indices
    ].add(sparse_bar.values)


def monoenergetic_interpolation_table_bar(grid_x, grid_nu, grid_Er, local_bar, table, database):
    """Explicit transpose of legacy ``get_Dij`` for one coefficient table.

    This mirrors the primal's C1 bicubic slice interpolation and its legacy
    small/mid/large radial polynomials.  It is table-only by construction:
    the dynamic state query coordinates are treated as primal values.
    """

    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    grid_er_internal = jnp.where(
        grid_x <= database.low_limit_r,
        jnp.log10(database.Er_lower_limit),
        jnp.log10(jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er / grid_x))),
    )

    def _surface_bar(surface_index, weight):
        er_values = jax.lax.dynamic_index_in_dim(database.Er_list, surface_index, axis=0, keepdims=False)
        surface_table = jax.lax.dynamic_index_in_dim(table, surface_index, axis=0, keepdims=False)
        local = _monoenergetic_slice_table_bar(
            database.nu_log, er_values, grid_nu_internal, grid_er_internal, weight * local_bar, surface_table
        )
        return jnp.zeros_like(table).at[surface_index].add(local)

    # Reproduce ``interpolation_small_r`` algebra exactly by evaluating its
    # three linear basis vectors.  This avoids a numerically different matrix
    # inverse in a transpose that is compared against the established VJP.
    r12, r22, r32 = database.r1**2, database.r2**2, database.r3**2
    r13, r23, r33 = database.r1**3, database.r2**3, database.r3**3
    xr2, xr3 = grid_x**2, grid_x**3
    denom_a = (r32-r22)/(r33-r23) - (r32-r12)/(r33-r13)
    denom_b = (r33-r23)/(r32-r22) - (r33-r13)/(r32-r12)
    small_a = jnp.asarray((
        1.0/(r33-r13)/denom_a,
        -1.0/(r33-r23)/denom_a,
        (1.0/(r33-r23)-1.0/(r33-r13))/denom_a,
    ))
    small_b = jnp.asarray((
        1.0/(r32-r12)/denom_b,
        -1.0/(r32-r22)/denom_b,
        (1.0/(r32-r22)-1.0/(r32-r12))/denom_b,
    ))
    small_weights = jnp.asarray((1.0, 0.0, 0.0)) + (xr2-r12)*small_a + (xr3-r13)*small_b

    index = jnp.argmax(jnp.where(grid_x - database.rho[1:-1] * database.a_b <= 0.0, grid_x - database.rho[1:-1] * database.a_b, -jnp.inf)) + 1
    mid_indices = index + jnp.asarray((-2, -1, 0, 1), dtype=jnp.int32)
    mid_radii = database.a_b * database.rho[mid_indices]
    mid_weights = jnp.asarray((
        (grid_x-mid_radii[1])*(grid_x-mid_radii[2])*(grid_x-mid_radii[3]) / ((mid_radii[0]-mid_radii[1])*(mid_radii[0]-mid_radii[2])*(mid_radii[0]-mid_radii[3])),
        (grid_x-mid_radii[0])*(grid_x-mid_radii[2])*(grid_x-mid_radii[3]) / ((mid_radii[1]-mid_radii[0])*(mid_radii[1]-mid_radii[2])*(mid_radii[1]-mid_radii[3])),
        (grid_x-mid_radii[0])*(grid_x-mid_radii[1])*(grid_x-mid_radii[3]) / ((mid_radii[2]-mid_radii[0])*(mid_radii[2]-mid_radii[1])*(mid_radii[2]-mid_radii[3])),
        (grid_x-mid_radii[0])*(grid_x-mid_radii[1])*(grid_x-mid_radii[2]) / ((mid_radii[3]-mid_radii[0])*(mid_radii[3]-mid_radii[1])*(mid_radii[3]-mid_radii[2])),
    ))
    large_radii = jnp.asarray((database.rnm3, database.rnm2, database.rnm1))
    large_weights = jnp.asarray((
        (grid_x-large_radii[1])*(grid_x-large_radii[2]) / ((large_radii[0]-large_radii[1])*(large_radii[0]-large_radii[2])),
        (grid_x-large_radii[0])*(grid_x-large_radii[2]) / ((large_radii[1]-large_radii[0])*(large_radii[1]-large_radii[2])),
        (grid_x-large_radii[0])*(grid_x-large_radii[1]) / ((large_radii[2]-large_radii[0])*(large_radii[2]-large_radii[1])),
    ))

    small = sum((_surface_bar(i, w) for i, w in zip((0, 1, 2), small_weights, strict=True)), jnp.zeros_like(table))
    mid = sum((_surface_bar(i, w) for i, w in zip(mid_indices, mid_weights, strict=True)), jnp.zeros_like(table))
    n_radius = table.shape[0]
    large = sum((_surface_bar(i, w) for i, w in zip((n_radius - 3, n_radius - 2, n_radius - 1), large_weights, strict=True)), jnp.zeros_like(table))
    return jnp.where(
        grid_x < database.r1_lim,
        small,
        jnp.where(grid_x < database.rmn2_lim, mid, large),
    )


@jit
def get_Dij_loger_no_r(grid_x, grid_nu, grid_Er, database):
    xg = jnp.zeros(3)
    grid_nu_internal = jnp.log10(jnp.maximum(1.0e-12, grid_nu))
    grid_Er_internal = jnp.log10(jnp.maximum(database.Er_lower_limit, jnp.abs(grid_Er)))
    I = jnp.identity(3)
    array = jnp.select(
        condlist=[
            grid_x < database.r1_lim,
            (grid_x >= database.r1_lim) & (grid_x < database.rmn2_lim),
            grid_x >= database.rmn2_lim,
        ],
        choicelist=[I.at[0].get(), I.at[1].get(), I.at[2].get()],
        default=0,
    )
    xg = (
        array.at[0].get() * interpolation_small_r(grid_x, grid_nu_internal, grid_Er_internal, database)
        + array.at[1].get() * interpolation_mid_r(grid_x, grid_nu_internal, grid_Er_internal, database)
        + array.at[2].get() * interpolation_large_r(grid_x, grid_nu_internal, grid_Er_internal, database)
    )
    return xg

