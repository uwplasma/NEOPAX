import dataclasses

import jax
import jax.numpy as jnp
from jaxtyping import Array


def _expand_constraint(value, dtype):
    return jnp.expand_dims(jnp.asarray(value, dtype=dtype), axis=-1)


@jax.jit
def _linear_inner_face_values(value, face_centers):
    cell_centers = 0.5 * (face_centers[1:] + face_centers[:-1])
    face_x = face_centers[1:-1]
    left_x = cell_centers[:-1]
    right_x = cell_centers[1:]
    weight = (face_x - left_x) / (right_x - left_x)
    return value[..., :-1] + weight * (value[..., 1:] - value[..., :-1])


@jax.jit
def _weno3_inner_face_values(value, left_face_value, right_face_value):
    """WENO3-like bounded reconstruction for interior faces.

    Notes
    -----
    - Operates on the last axis as the radial cell axis.
    - Uses boundary face values as ghost information.
    - Clamps reconstructed face values to local neighbor bounds to avoid overshoots.
    """
    vpad = jnp.concatenate([left_face_value, value, right_face_value], axis=-1)

    v_im1 = vpad[..., :-3]
    v_i = vpad[..., 1:-2]
    v_ip1 = vpad[..., 2:-1]
    v_ip2 = vpad[..., 3:]

    eps = jnp.asarray(1.0e-12, dtype=value.dtype)

    # Left-biased reconstruction to i+1/2.
    q0 = -0.5 * v_im1 + 1.5 * v_i
    q1 = 0.5 * v_i + 0.5 * v_ip1
    beta0 = jnp.square(v_i - v_im1)
    beta1 = jnp.square(v_ip1 - v_i)
    alpha0 = (1.0 / 3.0) / jnp.square(beta0 + eps)
    alpha1 = (2.0 / 3.0) / jnp.square(beta1 + eps)
    wsum = alpha0 + alpha1
    left_rec = (alpha0 / wsum) * q0 + (alpha1 / wsum) * q1

    # Right-biased reconstruction to i+1/2.
    p0 = 0.5 * v_i + 0.5 * v_ip1
    p1 = 1.5 * v_ip1 - 0.5 * v_ip2
    beta0_r = jnp.square(v_ip1 - v_i)
    beta1_r = jnp.square(v_ip2 - v_ip1)
    alpha0_r = (2.0 / 3.0) / jnp.square(beta0_r + eps)
    alpha1_r = (1.0 / 3.0) / jnp.square(beta1_r + eps)
    wsum_r = alpha0_r + alpha1_r
    right_rec = (alpha0_r / wsum_r) * p0 + (alpha1_r / wsum_r) * p1

    inner = 0.5 * (left_rec + right_rec)
    lower = jnp.minimum(v_i, v_ip1)
    upper = jnp.maximum(v_i, v_ip1)
    return jnp.clip(inner, lower, upper)


@jax.jit
def _quadratic_face_derivative(x_eval, x0, x1, x2, y0, y1, y2):
    l0_prime = (2.0 * x_eval - x1 - x2) / ((x0 - x1) * (x0 - x2))
    l1_prime = (2.0 * x_eval - x0 - x2) / ((x1 - x0) * (x1 - x2))
    l2_prime = (2.0 * x_eval - x0 - x1) / ((x2 - x0) * (x2 - x1))
    return y0 * l0_prime + y1 * l1_prime + y2 * l2_prime


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CellVariable:
    """Finite-volume radial profile with face-grid geometry and boundary constraints."""

    value: Array
    face_centers: Array
    left_face_constraint: Array | None = None
    right_face_constraint: Array | None = None
    left_face_grad_constraint: Array | None = None
    right_face_grad_constraint: Array | None = None

    @property
    def cell_centers(self):
        return 0.5 * (self.face_centers[1:] + self.face_centers[:-1])

    @property
    def cell_widths(self):
        return jnp.diff(self.face_centers)

    def left_face_value(self):
        if self.left_face_constraint is not None:
            return _expand_constraint(self.left_face_constraint, self.value.dtype)
        if self.left_face_grad_constraint is not None:
            grad = _expand_constraint(self.left_face_grad_constraint, self.value.dtype)
            return self.value[..., :1] - grad * self.cell_widths[0] / 2.0
        return self.value[..., :1]

    @property
    def right_face_value(self):
        if self.right_face_constraint is not None:
            return _expand_constraint(self.right_face_constraint, self.value.dtype)
        if self.right_face_grad_constraint is not None:
            grad = _expand_constraint(self.right_face_grad_constraint, self.value.dtype)
            return self.value[..., -1:] + grad * self.cell_widths[-1] / 2.0
        return self.value[..., -1:]

    def face_value(self, reconstruction: str = "linear"):
        if self.value.shape[-1] == 1:
            return jnp.concatenate([self.left_face_value(), self.right_face_value], axis=-1)

        if reconstruction == "weno3":
            inner = _weno3_inner_face_values(
                self.value,
                self.left_face_value(),
                self.right_face_value,
            )
        else:
            inner = _linear_inner_face_values(self.value, self.face_centers)

        return jnp.concatenate([self.left_face_value(), inner, self.right_face_value], axis=-1)

    def face_grad(self):
        n_cell = self.value.shape[-1]
        half_width_left = self.cell_widths[0] / 2.0
        half_width_right = self.cell_widths[-1] / 2.0

        if self.left_face_grad_constraint is not None:
            left_grad = jnp.asarray(self.left_face_grad_constraint, dtype=self.value.dtype)
        else:
            left_face = self.left_face_value()[..., 0]
            left_grad = (self.value[..., 0] - left_face) / half_width_left

        if self.right_face_grad_constraint is not None:
            right_grad = jnp.asarray(self.right_face_grad_constraint, dtype=self.value.dtype)
        else:
            right_face = self.right_face_value[..., 0]
            right_grad = (right_face - self.value[..., -1]) / half_width_right

        if n_cell == 1:
            return jnp.stack([left_grad, right_grad], axis=-1)

        x = self.cell_centers
        if n_cell == 2:
            slope = (self.value[..., 1] - self.value[..., 0]) / (x[1] - x[0])
            return jnp.stack([left_grad, slope, right_grad], axis=-1)

        x_eval = self.face_centers[1:-1]
        inner = _quadratic_face_derivative(
            x_eval[:-1],
            x[:-2],
            x[1:-1],
            x[2:],
            self.value[..., :-2],
            self.value[..., 1:-1],
            self.value[..., 2:],
        )
        last_inner = _quadratic_face_derivative(
            self.face_centers[-2],
            x[-2],
            x[-1],
            self.face_centers[-1],
            self.value[..., -2],
            self.value[..., -1],
            self.right_face_value[..., 0],
        )
        return jnp.concatenate(
            [
                jnp.expand_dims(left_grad, axis=-1),
                inner,
                jnp.expand_dims(last_inner, axis=-1),
                jnp.expand_dims(right_grad, axis=-1),
            ],
            axis=-1,
        )

    def grad(self):
        face_values = self.face_value()
        return jnp.diff(face_values, axis=-1) / jnp.diff(self.face_centers)


def make_profile_cell_variable(
    profile,
    face_centers,
    *,
    left_face_constraint=None,
    right_face_constraint=None,
    left_face_grad_constraint=0.0,
    right_face_grad_constraint=None,
):
    return CellVariable(
        value=jnp.atleast_1d(jnp.asarray(profile)),
        face_centers=jnp.asarray(face_centers),
        left_face_constraint=left_face_constraint,
        right_face_constraint=right_face_constraint,
        left_face_grad_constraint=left_face_grad_constraint,
        right_face_grad_constraint=right_face_grad_constraint,
    )


def get_profile_gradient(
    profile,
    r_grid,
    r_grid_half,
    dr,
    right_face_constraint=None,
    right_face_grad_constraint=None,
):
    del r_grid, dr
    if right_face_constraint is None and right_face_grad_constraint is None:
        right_face_constraint = jnp.asarray(profile)[-1]

    cell_var = make_profile_cell_variable(
        profile,
        r_grid_half,
        right_face_constraint=right_face_constraint,
        right_face_grad_constraint=right_face_grad_constraint,
        left_face_grad_constraint=jnp.asarray(0.0, dtype=jnp.asarray(profile).dtype),
    )
    return cell_var.grad()


def get_gradient_density(
    density,
    r_grid,
    r_grid_half,
    dr,
    right_face_constraint=None,
    right_face_grad_constraint=None,
):
    return get_profile_gradient(
        density,
        r_grid,
        r_grid_half,
        dr,
        right_face_constraint=right_face_constraint,
        right_face_grad_constraint=right_face_grad_constraint,
    )


def get_gradient_temperature(
    temperature,
    r_grid,
    r_grid_half,
    dr,
    right_face_constraint=None,
    right_face_grad_constraint=None,
):
    return get_profile_gradient(
        temperature,
        r_grid,
        r_grid_half,
        dr,
        right_face_constraint=right_face_constraint,
        right_face_grad_constraint=right_face_grad_constraint,
    )