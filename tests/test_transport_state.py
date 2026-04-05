import jax.numpy as jnp
from NEOPAX._transport_solver import TransportState, FVMScheme, BoundaryCondition, ComposedTransportModel

def test_transport_state_pytree():
    # Create a dummy state
    density = jnp.ones((2, 8))
    temperature = jnp.ones((2, 8)) * 2.0
    Er = jnp.zeros(8)
    state = TransportState(density=density, temperature=temperature, Er=Er)
    # Check shapes
    assert state.density.shape == (2, 8)
    assert state.temperature.shape == (2, 8)
    assert state.Er.shape == (8,)

def test_bc_and_fvm():
    # Dummy field with required attributes
    class DummyField:
        dr = 1.0
        Vprime = jnp.ones(8)
        Vprime_half = jnp.ones(8)
    field = DummyField()
    fvm = FVMScheme(field)
    arr = jnp.arange(8.)
    arr_ext = fvm.extend_with_ghosts(arr)
    arr_bc = fvm.apply_dirichlet_ghosts(arr)
    assert arr_ext.shape == (10,)
    assert arr_bc.shape == (10,)
    bc = BoundaryCondition()
    arr_bc2 = bc.apply(arr)
    assert arr_bc2.shape == (10,)

def test_composed_model():
    # Compose dummy models
    class DummySource:
        def compute(self, state):
            return state
    field = type('DummyField', (), {'dr': 1.0, 'Vprime': jnp.ones(8), 'Vprime_half': jnp.ones(8)})()
    fvm = FVMScheme(field)
    bc = BoundaryCondition()
    model = ComposedTransportModel(fvm, DummySource(), bc)
    density = jnp.ones((2, 8))
    temperature = jnp.ones((2, 8)) * 2.0
    Er = jnp.zeros(8)
    state = TransportState(density=density, temperature=temperature, Er=Er)
    out = model(state)
    assert isinstance(out, TransportState)
    assert out.density.shape == (10,)
    assert out.temperature.shape == (10,)
    assert out.Er.shape == (10,)
