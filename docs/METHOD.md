# Method

The completed Riemann function is

\[
\Xi(s)=\tfrac12s(s-1)\pi^{-s/2}\Gamma(s/2)\zeta(s).
\]

Define

\[
\Phi(u)=\Xi(1/2+i\sqrt u)/\Xi(1/2),\qquad R(u)=-\Phi'(u)/\Phi(u).
\]

The code extracts Taylor coefficients of `Xi` by an arbitrary-precision Cauchy contour, forms `Phi`, and performs formal power-series division to obtain moments `r_n` of `R`. The moment inner product

\[
\langle p,q\rangle=\sum_{i,j}p_iq_jr_{i+j}
\]

generates orthonormal polynomials and the Jacobi recurrence. Finite truncations are nested prefixes of one recurrence. Their positive eigenvalues are interpreted as `x_j≈1/gamma_j²`.

The GUE protocol compares consecutive truncations, freezes a drift tolerance, selects the longest consecutive stable block, and evaluates nearest-neighbour spacings using three unfolding methods and unfolding-free spacing ratios.
