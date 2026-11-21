"""Independent extended-precision VMEC native inputs for geometry FD checks.

Keep the displacement in host longdouble through the existing forward field
algebra. No differentiation is used here. Recenter its finite increments on
the ordinary CUDA baseline so a host/device primal offset is not interpreted
as a parameter perturbation. This changes benchmark endpoint arithmetic only.
"""
from __future__ import annotations

import dataclasses
from types import FunctionType, SimpleNamespace

import numpy as np


class _Array(np.ndarray):
    @property
    def at(self):
        return _Update(self)


class _Update:
    def __init__(self, array, index=None):
        self.array, self.index = array, index

    def __getitem__(self, index):
        return _Update(self.array, index)

    def set(self, value):
        result = self.array.copy()
        result[self.index] = value
        return result

    def multiply(self, value):
        result = self.array.copy()
        result[self.index] *= value
        return result


class _Numpy:
    float64 = np.longdouble

    def __getattr__(self, name):
        value = getattr(np, name)
        if not callable(value) or isinstance(value, type):
            return value

        def call(*args, **kwargs):
            kwargs.pop('precision', None)  # JAX dot-product precision option.
            if 'dtype' in kwargs and np.dtype(kwargs['dtype']).kind == 'f':
                kwargs['dtype'] = np.longdouble
            result = value(*args, **kwargs)
            if isinstance(result, np.ndarray):
                if result.dtype.kind == 'f':
                    result = result.astype(np.longdouble, copy=False)
                result = result.view(_Array)
            return result
        return call


def _cubic(query, nodes, values, *, method, extrap):
    """Interpax's local cubic Hermite rule, evaluated in extended precision."""
    if method != 'cubic' or extrap is not True:
        raise ValueError('Extended VMEC FD requires the existing cubic/extrap rule')
    slope = np.diff(values, axis=0) / np.diff(nodes).reshape(
        (-1,) + (1,) * (values.ndim - 1))
    derivative = np.concatenate((slope[:1], (slope[:-1] + slope[1:]) / 2, slope[-1:]))
    i = np.clip(np.searchsorted(nodes, query, side='right'), 1, len(nodes) - 1)
    dx = nodes[i] - nodes[i-1]
    t = (query - nodes[i-1]) / dx
    return ((2*t**3 - 3*t**2 + 1) * values[i-1]
            + (-2*t**3 + 3*t**2) * values[i]
            + (t**3 - 2*t**2 + t) * dx * derivative[i-1]
            + (t**3 - t**2) * dx * derivative[i])


def extended_forward(function, import_vmec_module):
    """Copy forward functions' namespaces; never mutate imported modules.

    The copied functions execute their original code with a NumPy namespace.
    Only plain functions referencing that same array namespace are copied;
    an unsupported JAX-only operation fails instead of calling an AD rule.
    """
    backend, cache = _Numpy(), {}

    class Module:
        def __init__(self, module):
            self.module = module

        def __getattr__(self, name):
            value = getattr(self.module, name)
            return copy(value) if isinstance(value, FunctionType) and 'jnp' in value.__globals__ else value

    def copy(fn):
        if fn in cache:
            return cache[fn]
        namespace = fn.__globals__.copy()
        result = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
        result.__kwdefaults__ = fn.__kwdefaults__
        cache[fn] = result
        for name in fn.__code__.co_names:
            value = namespace.get(name)
            if isinstance(value, FunctionType) and 'jnp' in value.__globals__:
                namespace[name] = copy(value)
        namespace.update(jnp=backend, interpax=SimpleNamespace(interp1d=_cubic),
            _import_vmec_module=lambda name: Module(import_vmec_module(name)))
        return result
    return copy(function)


def recenter(baseline, reference, endpoint):
    """Preserve the ordinary baseline while correcting only finite increments."""
    result = {}
    for field in dataclasses.fields(baseline):
        base = getattr(baseline, field.name)
        if base is None or np.asarray(base).dtype.kind != 'f':
            continue
        a, b = np.asarray(getattr(reference, field.name)), np.asarray(getattr(endpoint, field.name))
        if a.shape != np.shape(base) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid extended native field: ' + field.name)
        corrected = np.asarray(base, dtype=np.longdouble) + (b - a)
        converted = np.asarray(corrected, dtype=np.asarray(base).dtype)
        result[field.name] = type(base)(converted) if isinstance(base, (float, np.floating)) else converted
    return dataclasses.replace(baseline, **result)


class NativePrecision:
    """Cache one high-precision baseline per set of requested native surfaces."""
    def __init__(self, geometry_module, state, direction):
        if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
            raise RuntimeError('Extended native FD needs a host with wider-than-float64 longdouble')
        self.original = geometry_module._traceable_vmec_surfaces_from_state
        self.forward = extended_forward(self.original, geometry_module._import_vmec_module)
        self.state = state
        self.direction = direction
        self.references = {}

    def remember(self, s_values, baseline):
        """Retain the exact surfaces used by the ordinary baseline scan."""
        self.references[tuple(s_values)] = baseline, None

    def evaluate(self, context, step, *, s_values):
        key = tuple(s_values)
        if key not in self.references:
            self.remember(key, self.original(context, self.state, s_values=key))
        baseline, reference = self.references[key]
        if reference is None:
            host = dataclasses.replace(self.state, **{
                f.name: np.asarray(getattr(self.state, f.name), dtype=np.longdouble)
                for f in dataclasses.fields(self.state)})
            reference = self.forward(context, host, s_values=key)
            self.references[key] = baseline, reference
        host = dataclasses.replace(self.state, **{
            f.name: np.asarray(getattr(self.state, f.name), dtype=np.longdouble)
                + np.longdouble(step) * np.asarray(getattr(self.direction, f.name), dtype=np.longdouble)
            for f in dataclasses.fields(self.state)})
        endpoint = self.forward(context, host, s_values=key)
        return tuple(recenter(a, b, c) for a, b, c in zip(baseline, reference, endpoint, strict=True))
