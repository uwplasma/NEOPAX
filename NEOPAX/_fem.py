import jax.numpy as jnp

from ._cell_variable import (
    CellVariable,
    get_gradient_density,
    get_gradient_temperature,
    get_profile_gradient,
    make_profile_cell_variable,
)


"""
_fem.py: Finite Volume/Element Methods for NEOPAX

This module provides reusable finite volume (FVM) and finite element (FEM) utilities for NEOPAX transport solvers.

- Cell/face averaging
- Flux computation at interfaces
- Conservative update routines
- Boundary condition helpers

Inspired by the modular structure in Torax.
"""

# Example: Compute cell-centered values from face values
def cell_centered_from_faces(face_values):
    """Average face values to cell centers."""
    return 0.5 * (face_values[:-1] + face_values[1:])

# Example: Compute face values from cell-centered values
def faces_from_cell_centered(cell_values):
    """Average cell values to faces (interfaces)."""
    return 0.5 * (jnp.concatenate([cell_values[:1], cell_values]) +
                  jnp.concatenate([cell_values, cell_values[-1:]]))

# Example: Compute conservative finite volume update
def conservative_update(flux, dx, Vprime=None, Vprime_half=None, source=None):
    """Update cell values using net flux and optional source, with optional volume weighting.
    If Vprime and Vprime_half are provided, use stellarator volume form:
        update = -1/(Vprime*dx) * (Vprime_half[1:] * flux[1:] - Vprime_half[:-1] * flux[:-1])
    Otherwise, defaults to simple FVM.

    Notes:
    - ``dx`` can be scalar (uniform spacing) or a per-cell array of shape ``(n_r,)``.
    - JAX broadcasting handles both modes.
    """
    if Vprime is not None and Vprime_half is not None:
        net_flux = Vprime_half[1:] * flux[1:] - Vprime_half[:-1] * flux[:-1]
        update = -net_flux / (Vprime * dx)
    else:
        net_flux = flux[1:] - flux[:-1]
        update = -net_flux / dx
    if source is not None:
        update = update + source
    return update

# Example: Dirichlet boundary condition for fluxes
def apply_dirichlet_flux(flux, left_bc, right_bc):
    """Set left/right boundary fluxes for Dirichlet BCs."""
    flux = flux.at[0].set(left_bc)
    flux = flux.at[-1].set(right_bc)
    return flux

# Neumann boundary condition for FVM
def apply_neumann_flux(flux, grad_value, dr, side='right'):
    """
    Set boundary face flux for a Neumann BC: dQ/dr = grad_value at the boundary.
    For the right (edge) boundary, set flux[-1] so that:
        (Q_edge - Q_interior)/dr = grad_value
        Q_edge = Q_interior + grad_value * dr
    """
    if side == 'right':
        Q_interior = flux[-2]  # last interior face value
        Q_edge = Q_interior + grad_value * dr
        flux = flux.at[-1].set(Q_edge)
    elif side == 'left':
        Q_interior = flux[1]  # first interior face value
        Q_edge = Q_interior - grad_value * dr
        flux = flux.at[0].set(Q_edge)
    return flux


# General Robin/decay boundary condition for FVM
def apply_robin_flux(flux, cell_value, decay_length, dr, side='right'):
    """
    Set boundary face flux for a Robin/decay BC: dQ/dr = -Q/decay_length at the boundary.
    For the right (edge) boundary, set flux[-1] so that:
        (Q_edge - Q_interior)/dr = -Q_edge/decay_length
    Solving for Q_edge:
        Q_edge = Q_interior / (1 + dr/decay_length)
    """
    if side == 'right':
        Q_interior = cell_value[-1]
        Q_edge = Q_interior / (1 + dr/decay_length)
        flux = flux.at[-1].set(Q_edge)
    elif side == 'left':
        Q_interior = cell_value[0]
        Q_edge = Q_interior / (1 + dr/decay_length)
        flux = flux.at[0].set(Q_edge)
    return flux

# Ghost cell boundary condition utilities
def set_dirichlet_ghosts(u, left_value, right_value):
    """Set Dirichlet BCs by assigning ghost cell values."""
    u = u.at[0].set(left_value)
    u = u.at[-1].set(right_value)
    return u

def set_neumann_ghosts(u, grad_left, grad_right, dr):
    """Set Neumann BCs by assigning ghost cell values using gradients and dr."""
    u = u.at[0].set(u[1] - grad_left * dr)
    u = u.at[-1].set(u[-2] + grad_right * dr)
    return u

# FVMScheme: Finite Volume utility class for modular FVM operations
class FVMScheme:
    def __init__(self, field):
        self.field = field
        self.dr = field.dr
        self.Vprime = field.Vprime
        self.Vprime_half = field.Vprime_half

    def conservative_update(self, flux):
        from ._fem import conservative_update
        return conservative_update(flux, self.dr, self.Vprime, self.Vprime_half)

    def extend_with_ghosts(self, arr):
        return jnp.concatenate([arr[:1], arr, arr[-1:]])

    def apply_dirichlet_ghosts(self, arr):
        from ._fem import set_dirichlet_ghosts
        arr_ext = self.extend_with_ghosts(arr)
        return set_dirichlet_ghosts(arr_ext, arr[0], arr[-1])

    def faces_from_cell_centered(self, arr):
        from ._fem import faces_from_cell_centered
        return faces_from_cell_centered(arr)