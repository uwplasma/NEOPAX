Mirror Geometry
===============

Mirror-related geometry quantities enter NEOPAX through the VMEC/Boozer
geometry path and through geometry objectives used in optimization workflows.

In transport runs, the geometry provides:

- radial volume factors
- magnetic-field normalization
- rotational transform
- metric-like factors used by electric-field and neoclassical calculations

In geometry optimization, mirror ratio and Boozer-space quantities can be used
as objective or diagnostic terms together with QI, max-J, aspect ratio, iota,
and transport objectives.

The exact set of available geometry objectives is exposed through the
optimization module and benchmark scripts.  The transport solver itself
consumes the resulting geometry through the same grid and volume-factor
interface used by ordinary forward runs.

