"""Pure layout checks for the opt-in compact joint-lowdot carry.

This file deliberately imports only JAX.  It must remain runnable without an
NTX system, a transport rollout, or any compilation/profile output.
"""

import jax
import jax.numpy as jnp


def test_compact_joint_prepared_carry_matches_leaf_scatter_mock():
    """Packed joint-support accumulation equals the original leaf scatter."""
    objective_count = 3
    radius_count = 5
    prepared = {
        "surface": jnp.zeros((radius_count, 2), dtype=jnp.float64),
        "geometry": jnp.zeros((radius_count, 2, 2), dtype=jnp.float64),
    }
    leaves, treedef = jax.tree_util.tree_flatten(prepared)
    local_shapes = tuple(leaf.shape[1:] for leaf in leaves)
    local_sizes = tuple(int(leaf.size // radius_count) for leaf in leaves)
    packed = jnp.zeros((objective_count, radius_count, sum(local_sizes)), dtype=jnp.float64)
    legacy = jax.tree_util.tree_map(
        lambda leaf: jnp.zeros((objective_count,) + leaf.shape, dtype=leaf.dtype),
        prepared,
    )

    for radius_index, scale in ((1, 1.0), (3, -0.5), (1, 2.0)):
        local_leaves = tuple(
            scale
            * jnp.arange(
                objective_count * int(jnp.prod(jnp.asarray(shape))), dtype=jnp.float64
            ).reshape((objective_count,) + shape)
            for shape in local_shapes
        )
        legacy = jax.tree_util.tree_map(
            lambda carry, local: carry.at[:, radius_index].add(local),
            legacy,
            treedef.unflatten(local_leaves),
        )
        packed = packed.at[:, radius_index].add(
            jnp.concatenate(
                tuple(jnp.reshape(local, (objective_count, -1)) for local in local_leaves),
                axis=1,
            )
        )

    compact_leaves = []
    offset = 0
    for leaf, shape, size in zip(leaves, local_shapes, local_sizes, strict=True):
        compact_leaves.append(
            jnp.reshape(
                packed[:, :, offset : offset + size],
                (objective_count, radius_count) + shape,
            ).astype(leaf.dtype)
        )
        offset += size
    compact = treedef.unflatten(compact_leaves)

    for compact_leaf, legacy_leaf in zip(
        jax.tree_util.tree_leaves(compact),
        jax.tree_util.tree_leaves(legacy),
        strict=True,
    ):
        assert jnp.array_equal(compact_leaf, legacy_leaf)
