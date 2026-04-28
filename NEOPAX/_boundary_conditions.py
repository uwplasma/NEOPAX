"""
Modular boundary condition (BC) abstraction for NEOPAX, inspired by torax.
"""
import dataclasses

import jax.numpy as jnp

class BoundaryCondition:
    def apply(self, flux, state=None, axis=True, edge=True, **kwargs):
        raise NotImplementedError

class DirichletBC(BoundaryCondition):
    def __init__(self, axis_value, edge_value):
        self.axis_value = axis_value
        self.edge_value = edge_value
    def apply(self, flux, state=None, axis=True, edge=True, **kwargs):
        if axis:
            flux = flux.at[0].set(self.axis_value)
        if edge:
            flux = flux.at[-1].set(self.edge_value)
        return flux

class NeumannBC(BoundaryCondition):
    def __init__(self, grad_axis, grad_edge, dr):
        self.grad_axis = grad_axis
        self.grad_edge = grad_edge
        self.dr = dr
    def apply(self, flux, state=None, axis=True, edge=True, **kwargs):
        if axis:
            flux = flux.at[0].set(flux[1] - self.grad_axis * self.dr)
        if edge:
            flux = flux.at[-1].set(flux[-2] + self.grad_edge * self.dr)
        return flux

class RobinBC(BoundaryCondition):
    def __init__(self, alpha_axis, beta_axis, gamma_axis, alpha_edge, beta_edge, gamma_edge, dr):
        self.alpha_axis = alpha_axis
        self.beta_axis = beta_axis
        self.gamma_axis = gamma_axis
        self.alpha_edge = alpha_edge
        self.beta_edge = beta_edge
        self.gamma_edge = gamma_edge
        self.dr = dr
    def apply(self, flux, state, axis=True, edge=True, **kwargs):
        # Robin: alpha*u + beta*du/dr = gamma
        if axis:
            u1 = state[1]
            u0 = (self.gamma_axis - self.beta_axis * (u1 - state[0]) / self.dr) / self.alpha_axis
            flux = flux.at[0].set(u0)
        if edge:
            uN_1 = state[-2]
            uN = (self.gamma_edge - self.beta_edge * (state[-1] - uN_1) / self.dr) / self.alpha_edge
            flux = flux.at[-1].set(uN)
        return flux

# Registry for runtime BC registration
BC_REGISTRY = {}

def register_bc(name, bc_class):
    BC_REGISTRY[name] = bc_class

def get_bc(name, *args, **kwargs):
    return BC_REGISTRY[name](*args, **kwargs)

# Register built-in BCs
register_bc('dirichlet', DirichletBC)
register_bc('neumann', NeumannBC)
register_bc('robin', RobinBC)


@dataclasses.dataclass(frozen=True, eq=False)
class BoundaryConditionModel:
    """Left/right radial BC for 1D arrays with optional species-wise values."""

    dr: float
    left_type: str = "dirichlet"
    right_type: str = "dirichlet"
    left_value: jnp.ndarray | None = None
    right_value: jnp.ndarray | None = None
    left_gradient: jnp.ndarray | None = None
    right_gradient: jnp.ndarray | None = None
    left_decay_length: jnp.ndarray | None = None
    right_decay_length: jnp.ndarray | None = None
    reference_profile: jnp.ndarray | None = None
    reference_profiles: jnp.ndarray | None = None

    @staticmethod
    def _as_jnp_or_none(value, species_names=None):
        if value is None:
            return None
        if isinstance(value, dict):
            if species_names is None:
                return jnp.asarray(list(value.values()))
            default = value.get("default", None)
            ordered = []
            for name in species_names:
                if name in value:
                    ordered.append(value[name])
                elif default is not None:
                    ordered.append(default)
                else:
                    raise ValueError(
                        f"Missing boundary value for species '{name}'. "
                        "Provide all species explicitly or add a 'default' entry."
                    )
            return jnp.asarray(ordered)
        return jnp.asarray(value)

    @staticmethod
    def _pick_for_row(value, row_index: int):
        if value is None:
            return None
        arr = jnp.asarray(value)
        if arr.ndim == 0:
            return arr
        idx = min(row_index, int(arr.shape[0]) - 1)
        return arr[idx]

    def _infer_gradient(self, arr: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if arr.shape[0] < 2:
            return jnp.asarray(0.0), jnp.asarray(0.0)
        left_grad = (arr[1] - arr[0]) / self.dr
        right_grad = (arr[-1] - arr[-2]) / self.dr
        return left_grad, right_grad

    def _infer_log_gradient_coeff(self, boundary_value: jnp.ndarray, boundary_grad: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.asarray(1e-12)
        coeff = jnp.abs(boundary_grad) / (jnp.abs(boundary_value) + eps)
        return jnp.where(jnp.isfinite(coeff), coeff, jnp.asarray(0.0))

    def _infer_decay_length(self, boundary_value: jnp.ndarray, boundary_grad: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.asarray(1e-12)
        coeff = self._infer_log_gradient_coeff(boundary_value, boundary_grad)
        return 1.0 / jnp.maximum(coeff, eps)

    def _apply_ghost_row(self, arr: jnp.ndarray, row_index: int, reference_row: jnp.ndarray | None) -> jnp.ndarray:
        arr_ext = jnp.concatenate([arr[:1], arr, arr[-1:]])
        ref = reference_row if reference_row is not None else arr

        left_type = self.left_type.lower()
        right_type = self.right_type.lower()

        left_value = self._pick_for_row(self.left_value, row_index)
        right_value = self._pick_for_row(self.right_value, row_index)
        left_grad = self._pick_for_row(self.left_gradient, row_index)
        right_grad = self._pick_for_row(self.right_gradient, row_index)
        left_decay = self._pick_for_row(self.left_decay_length, row_index)
        right_decay = self._pick_for_row(self.right_decay_length, row_index)

        inferred_left_grad, inferred_right_grad = self._infer_gradient(ref)

        if left_type == "dirichlet":
            lv = ref[0] if left_value is None else left_value
            arr_ext = arr_ext.at[0].set(lv)
        elif left_type == "neumann":
            lg = inferred_left_grad if left_grad is None else left_grad
            arr_ext = arr_ext.at[0].set(arr_ext[1] - lg * self.dr)
        elif left_type == "robin":
            lv = ref[0]
            lg = inferred_left_grad if left_grad is None else left_grad
            ll = self._infer_decay_length(lv, lg) if left_decay is None else left_decay
            robin_left_grad = lv / (ll + 1e-12)
            arr_ext = arr_ext.at[0].set(arr_ext[1] - robin_left_grad * self.dr)
        else:
            raise ValueError(f"Unsupported left BC type: {self.left_type}")

        if right_type == "dirichlet":
            rv = ref[-1] if right_value is None else right_value
            arr_ext = arr_ext.at[-1].set(rv)
        elif right_type == "neumann":
            rg = inferred_right_grad if right_grad is None else right_grad
            arr_ext = arr_ext.at[-1].set(arr_ext[-2] + rg * self.dr)
        elif right_type == "robin":
            rv = ref[-1]
            rg = inferred_right_grad if right_grad is None else right_grad
            rl = self._infer_decay_length(rv, rg) if right_decay is None else right_decay
            robin_right_grad = -rv / (rl + 1e-12)
            arr_ext = arr_ext.at[-1].set(arr_ext[-2] + robin_right_grad * self.dr)
        else:
            raise ValueError(f"Unsupported right BC type: {self.right_type}")

        return arr_ext

    def apply_ghost(self, arr: jnp.ndarray) -> jnp.ndarray:
        ref = self.reference_profile if self.reference_profile is not None else arr
        return self._apply_ghost_row(arr, row_index=0, reference_row=ref)

    def apply_ghost_all(self, arr2d: jnp.ndarray) -> jnp.ndarray:
        out = []
        for i in range(arr2d.shape[0]):
            ref_row = None
            if self.reference_profiles is not None:
                ref_row = self.reference_profiles[i]
            elif self.reference_profile is not None:
                ref_row = self.reference_profile
            out.append(self._apply_ghost_row(arr2d[i], row_index=i, reference_row=ref_row))
        return jnp.stack(out, axis=0)


def build_boundary_condition_model(
    bc_cfg: dict,
    dr: float,
    reference_profile=None,
    reference_profiles=None,
    species_names=None,
):
    left_cfg = bc_cfg.get("left", {}) if isinstance(bc_cfg, dict) else {}
    right_cfg = bc_cfg.get("right", {}) if isinstance(bc_cfg, dict) else {}

    return BoundaryConditionModel(
        dr=dr,
        left_type=str(left_cfg.get("type", "dirichlet")),
        right_type=str(right_cfg.get("type", "dirichlet")),
        left_value=BoundaryConditionModel._as_jnp_or_none(left_cfg.get("value"), species_names=species_names),
        right_value=BoundaryConditionModel._as_jnp_or_none(right_cfg.get("value"), species_names=species_names),
        left_gradient=BoundaryConditionModel._as_jnp_or_none(left_cfg.get("gradient"), species_names=species_names),
        right_gradient=BoundaryConditionModel._as_jnp_or_none(right_cfg.get("gradient"), species_names=species_names),
        left_decay_length=BoundaryConditionModel._as_jnp_or_none(left_cfg.get("decay_length"), species_names=species_names),
        right_decay_length=BoundaryConditionModel._as_jnp_or_none(right_cfg.get("decay_length"), species_names=species_names),
        reference_profile=None if reference_profile is None else jnp.asarray(reference_profile),
        reference_profiles=None if reference_profiles is None else jnp.asarray(reference_profiles),
    )


def _as_like_template(value, template):
    """Broadcast or resize scalar/species-wise values to match ``template`` shape."""
    template_arr = jnp.asarray(template)
    arr = jnp.asarray(value, dtype=template_arr.dtype)

    if template_arr.ndim == 0:
        return arr.reshape(-1)[0]

    if arr.ndim == 0:
        return jnp.broadcast_to(arr, template_arr.shape)

    if arr.shape[0] == template_arr.shape[0]:
        return arr

    if arr.shape[0] > template_arr.shape[0]:
        return arr[:template_arr.shape[0]]

    pad_len = template_arr.shape[0] - arr.shape[0]
    return jnp.pad(arr, (0, pad_len), mode="edge")


def _boundary_cell_spacing(face_centers, *, side: str):
    face_centers = jnp.asarray(face_centers)
    n_faces = face_centers.shape[0]
    if n_faces >= 4:
        cell_centers = 0.5 * (face_centers[1:] + face_centers[:-1])
        if side == "right":
            return cell_centers[-1] - cell_centers[-2]
        return cell_centers[1] - cell_centers[0]
    if side == "right":
        return face_centers[-1] - face_centers[-2]
    return face_centers[1] - face_centers[0]


def right_constraints_from_bc_model(bc_model, default_value, profile=None, face_centers=None):
    """Translate right BC model to ``CellVariable`` right-face constraints.

    Returns
    -------
    tuple
        ``(right_face_constraint, right_face_grad_constraint)`` where one of the
        entries is typically ``None`` and the other is a JAX array.
    """
    default_arr = jnp.asarray(default_value)
    zeros_like_default = jnp.zeros_like(default_arr)

    if bc_model is None:
        return None, zeros_like_default

    right_type = str(getattr(bc_model, "right_type", "dirichlet")).strip().lower()
    right_value = getattr(bc_model, "right_value", None)
    right_gradient = getattr(bc_model, "right_gradient", None)
    right_decay = getattr(bc_model, "right_decay_length", None)

    if profile is not None and face_centers is not None:
        prof = jnp.asarray(profile)
        dx = _boundary_cell_spacing(face_centers, side="right")
        if prof.ndim == 1:
            prof = prof[None, :]
        u_im1 = prof[:, -1]
        u_im2 = prof[:, -2] if prof.shape[-1] >= 2 else prof[:, -1]

        if right_type == "dirichlet":
            rv = default_arr if right_value is None else _as_like_template(right_value, default_arr)
            rg = (3.0 * rv - 4.0 * u_im1 + u_im2) / (2.0 * dx)
            return rv, rg

        if right_type == "neumann":
            rg = zeros_like_default if right_gradient is None else _as_like_template(right_gradient, default_arr)
            rv = (4.0 * u_im1 - u_im2 + 2.0 * dx * rg) / 3.0
            return rv, rg

        if right_type == "robin":
            decay = (
                jnp.ones_like(default_arr)
                if right_decay is None
                else _as_like_template(right_decay, default_arr)
            )
            rv = (4.0 * u_im1 - u_im2) / (3.0 + 2.0 * dx / (decay + 1e-12))
            rg = -rv / (decay + 1e-12)
            return rv, rg

    if right_type == "dirichlet":
        rv = default_arr if right_value is None else _as_like_template(right_value, default_arr)
        return rv, None

    if right_type == "neumann":
        rg = zeros_like_default if right_gradient is None else _as_like_template(right_gradient, default_arr)
        return None, rg

    if right_type == "robin":
        rv = default_arr
        decay = (
            jnp.zeros_like(default_arr)
            if right_decay is None
            else _as_like_template(right_decay, default_arr)
        )
        robin_grad = -rv / (decay + 1e-12)
        return None, robin_grad

    raise ValueError(f"Unsupported right BC type: {right_type}")


def left_constraints_from_bc_model(bc_model, default_value, profile=None, face_centers=None):
    """Translate left BC model to ``CellVariable`` left-face constraints."""
    default_arr = jnp.asarray(default_value)
    zeros_like_default = jnp.zeros_like(default_arr)

    if bc_model is None:
        return None, zeros_like_default

    left_type = str(getattr(bc_model, "left_type", "dirichlet")).strip().lower()
    left_value = getattr(bc_model, "left_value", None)
    left_gradient = getattr(bc_model, "left_gradient", None)
    left_decay = getattr(bc_model, "left_decay_length", None)

    if profile is not None and face_centers is not None:
        prof = jnp.asarray(profile)
        dx = _boundary_cell_spacing(face_centers, side="left")
        if prof.ndim == 1:
            prof = prof[None, :]
        u_ip1 = prof[:, 0]
        u_ip2 = prof[:, 1] if prof.shape[-1] >= 2 else prof[:, 0]

        if left_type == "dirichlet":
            lv = default_arr if left_value is None else _as_like_template(left_value, default_arr)
            lg = (-3.0 * lv + 4.0 * u_ip1 - u_ip2) / (2.0 * dx)
            return lv, lg

        if left_type == "neumann":
            lg = zeros_like_default if left_gradient is None else _as_like_template(left_gradient, default_arr)
            lv = (4.0 * u_ip1 - u_ip2 - 2.0 * dx * lg) / 3.0
            return lv, lg

        if left_type == "robin":
            decay = (
                jnp.ones_like(default_arr)
                if left_decay is None
                else _as_like_template(left_decay, default_arr)
            )
            lv = (4.0 * u_ip1 - u_ip2) / (3.0 - 2.0 * dx / (decay + 1e-12))
            lg = lv / (decay + 1e-12)
            return lv, lg

    if left_type == "dirichlet":
        lv = default_arr if left_value is None else _as_like_template(left_value, default_arr)
        return lv, None

    if left_type == "neumann":
        lg = zeros_like_default if left_gradient is None else _as_like_template(left_gradient, default_arr)
        return None, lg

    if left_type == "robin":
        lv = default_arr
        decay = (
            jnp.zeros_like(default_arr)
            if left_decay is None
            else _as_like_template(left_decay, default_arr)
        )
        robin_grad = lv / (decay + 1e-12)
        return None, robin_grad

    raise ValueError(f"Unsupported left BC type: {left_type}")


def apply_cell_centered_boundary_state(profile, bc_model, face_centers):
    """Project cell-centered profiles so boundary cells satisfy the configured BCs."""
    if bc_model is None:
        return jnp.asarray(profile)

    prof = jnp.asarray(profile)
    squeeze = prof.ndim == 1
    if squeeze:
        prof = prof[None, :]

    dx_left = _boundary_cell_spacing(face_centers, side="left")
    dx_right = _boundary_cell_spacing(face_centers, side="right")
    out = prof

    left_type = str(getattr(bc_model, "left_type", "dirichlet")).strip().lower()
    left_default = prof[:, 0]
    left_value = None if getattr(bc_model, "left_value", None) is None else _as_like_template(getattr(bc_model, "left_value"), left_default)
    left_gradient = None if getattr(bc_model, "left_gradient", None) is None else _as_like_template(getattr(bc_model, "left_gradient"), left_default)
    left_decay = None if getattr(bc_model, "left_decay_length", None) is None else _as_like_template(getattr(bc_model, "left_decay_length"), left_default)
    u1 = prof[:, 1] if prof.shape[1] >= 2 else prof[:, 0]
    u2 = prof[:, 2] if prof.shape[1] >= 3 else u1
    if left_type == "dirichlet":
        left_cell = left_default if left_value is None else left_value
        out = out.at[:, 0].set(left_cell)
    elif left_type == "neumann":
        lg = jnp.zeros_like(left_default) if left_gradient is None else left_gradient
        left_cell = (4.0 * u1 - u2 - 2.0 * dx_left * lg) / 3.0
        out = out.at[:, 0].set(left_cell)
    elif left_type == "robin":
        decay = jnp.ones_like(left_default) if left_decay is None else left_decay
        left_cell = (4.0 * u1 - u2) / (3.0 - 2.0 * dx_left / (decay + 1.0e-12))
        out = out.at[:, 0].set(left_cell)

    right_type = str(getattr(bc_model, "right_type", "dirichlet")).strip().lower()
    right_default = prof[:, -1]
    right_value = None if getattr(bc_model, "right_value", None) is None else _as_like_template(getattr(bc_model, "right_value"), right_default)
    right_gradient = None if getattr(bc_model, "right_gradient", None) is None else _as_like_template(getattr(bc_model, "right_gradient"), right_default)
    right_decay = None if getattr(bc_model, "right_decay_length", None) is None else _as_like_template(getattr(bc_model, "right_decay_length"), right_default)
    u_nm1 = prof[:, -2] if prof.shape[1] >= 2 else prof[:, -1]
    u_nm2 = prof[:, -3] if prof.shape[1] >= 3 else u_nm1
    if right_type == "dirichlet":
        right_cell = right_default if right_value is None else right_value
        out = out.at[:, -1].set(right_cell)
    elif right_type == "neumann":
        rg = jnp.zeros_like(right_default) if right_gradient is None else right_gradient
        right_cell = (4.0 * u_nm1 - u_nm2 + 2.0 * dx_right * rg) / 3.0
        out = out.at[:, -1].set(right_cell)
    elif right_type == "robin":
        decay = jnp.ones_like(right_default) if right_decay is None else right_decay
        right_cell = (4.0 * u_nm1 - u_nm2) / (3.0 + 2.0 * dx_right / (decay + 1.0e-12))
        out = out.at[:, -1].set(right_cell)

    return out[0] if squeeze else out
