# Momentum-conserving neoclassical correction

## Provenance

The momentum correction follows the method introduced in

> M. Taguchi, *A method for calculating neoclassical transport
> coefficients with momentum conserving collision operator*,
> *Physics of Fluids B* **4**(11), 3638--3643 (1992).

The implementation is a three-Sonine-moment version of that construction.
It was matched term-for-term to NTSSfusion's
`Neo-DKES/taguchi/corrected_neoclassical_fluxes.f90`, whose header states
that it is the Taguchi method extended to higher Sonine order.  The direct
linear solve used there is LAPACK `DGESV`; NEOPAX uses a dense direct solve
for the corresponding finite block system.

## Purpose

The underlying monoenergetic coefficients are obtained with a pitch-angle
scattering collision operator.  That operator alone does not conserve
interspecies parallel momentum.  The correction reconstructs the missing
field-particle part by solving for three parallel Sonine moments per kinetic
species.  It produces consistent corrections to the parallel flow and to
the radial particle and heat fluxes.

For species indices \(a,b\in\{1,\ldots,N_s\}\) and Sonine indices
\(j,k\in\{0,1,2\}\), write the unknown moments as
\(u_{a j}\).  NEOPAX solves the dense \(3N_s\)-dimensional system

\[
  \sum_{b=1}^{N_s}\sum_{k=0}^{2}
      \mathcal{M}_{a j,b k}u_{b k}=r_{a j}.
\]

The code builds the \(3\times3\) blocks in the same diagonal/off-diagonal
form as Taguchi/NTSSfusion:

\[
\mathcal{M}_{a b} =
\begin{cases}
I - f_a\bigl(C_a^{T}S_a+E_a\bigr)\odot W,& a=b,\\
-f_a\bigl(C_a^{T}C^{N}_{ab}/\tau_{ab}\bigr)\odot W,& a\ne b,
\end{cases}
\qquad
f_a=\frac{2}{v_{\mathrm{th},a}^{2}\langle B^2/B_0^2\rangle}.
\]

Here \(W\) is the fixed Sonine-expansion matrix, \(C_a,E_a\) are built
from the monoenergetic \(L_{ij},E_{ij}\) coefficients, \(S_a\) is the
species sum of collision moments, and \(C^N_{ab}/\tau_{ab}\) is the
interspecies field-particle coupling.  The important implementation detail
is that the predicate \(a=b\) is a **species** equality test.  It is not an
index into the \(3\times3\) Sonine identity.  The latter was the four-species
bug: species index 3 was inadvertently aliased to a Sonine index by
out-of-bounds indexing.

The corrected parallel-flow quantity used for bootstrap current is

\[
 U_{\parallel,a}=n_a u_{a0},
 \qquad
 J_{\parallel}=\sum_a q_a U_{\parallel,a},
\]

with the final unit conversion performed in the bootstrap-current consumer.
The \(u_{a0}\) result is unaffected by the additional radial-flux terms
below.

## Radial particle and heat corrections

Define \(g_{na}=\partial_r\ln n_a\),
\(g_{Ta}=\partial_r\ln T_a\), and \(Z_a=q_a/e\) for this derivation.  For
each pair \(a\ne b\), the four NTSSfusion/Taguchi sums are

\[
\begin{aligned}
 A^{(1)}_{ab} &=
 \left[g_{na}+g_{Ta}
 -\frac{Z_a}{Z_b}\frac{T_b}{T_a}(g_{nb}+g_{Tb})\right]
 \frac{C^M_{ab,00}}{\tau_{ab}},\\
 A^{(2)}_{ab} &=
 \left[g_{na}+g_{Ta}
 -\frac{Z_a}{Z_b}\frac{T_b}{T_a}(g_{nb}+g_{Tb})\right]
 \frac{\tfrac52 C^M_{ab,00}-C^M_{ab,10}}{\tau_{ab}},\\
 A^{(3)}_{ab} &=
 \frac{Z_a}{Z_b}\frac{T_b}{T_a}g_{Tb}
 \frac{C^N_{ab,01}}{\tau_{ab}},\\
 A^{(4)}_{ab} &=
 \frac{Z_a}{Z_b}\frac{T_b}{T_a}g_{Tb}
 \frac{\tfrac52 C^N_{ab,01}-C^N_{ab,11}}{\tau_{ab}}.
\end{aligned}
\]

Let \(\mathrm{ADD}_\ell=\sum_{b\ne a}A^{(\ell)}_{ab}\).  With the
thermodynamic forces \(X_{1a}\), \(X_{2a}\), the collision sum \(S_a\),
and the pitch-angle-scattering moments \(\nu_{a,k}\), NEOPAX applies the
added *velocity-normalized* radial flux terms

\[
\begin{aligned}
 \delta\!\left(\frac{\Gamma_a}{n_a}\right)
 &=P_a\left[
 \mathrm{ADD}_1-g_{Ta}S_{a,01}-\mathrm{ADD}_3
 +\frac{X_{1a}\nu_{a,0}+X_{2a}\nu_{a,1}}{3/2}
 \right],\\
 \delta\!\left(\frac{Q_a}{n_aT_a}\right)
 &=P_a\left[
 \mathrm{ADD}_2-g_{Ta}\left(\frac52S_{a,01}-S_{a,11}\right)-\mathrm{ADD}_4
 +\frac{X_{1a}\nu_{a,1}+X_{2a}\nu_{a,2}}{3/2}
 \right],\\
 P_a&=\frac{m_a T_a\,G_{\rm PS}}{(q_a B_0)^2}.
\end{aligned}
\]

In the NEOPAX implementation \(T_a\) is converted from keV to joules in
\(P_a\), and `G_PS` is the Pfirsch--Schl\u00fcter geometric factor.  The
outer factors \(n_a\) and \(n_aT_a\) are applied afterwards when forming
\(\Gamma_a\) and \(Q_a\).

## Scope in NEOPAX

`evaluate_momentum_corrected_fluxes` now computes all three corrected
quantities \((\Gamma,Q,U_\parallel)\).  The existing transport benchmark
still uses its ordinary radial-flux path; it uses the corrected
\(U_\parallel\) only for the bootstrap-current objective.  Routing the
corrected \(\Gamma,Q\) into the transport evolution would be a separate
forward-and-reverse mode, because it changes the Radau RHS and therefore its
adjoint.

## Regression checks

The current tests cover:

1. four-species diagonal/off-diagonal blocks using species equality;
2. a square \(3N_s\times3N_s\) correction matrix for the wHe case;
3. four-species corrected flux evaluation; and
4. direct agreement of the \(\mathrm{ADD}_{1\ldots4}\) radial terms with
   the Taguchi/NTSSfusion formulas above.

