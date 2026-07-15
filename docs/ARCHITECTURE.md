# Architecture

The repository implements the deterministic chain

`Xi -> Phi -> primitive resolvent R -> moments -> Stieltjes recursion -> Jacobi operator -> spectral nodes`.

## Generator and operator

The multiplication operator `M_x f=x f` on the conjectural resolvent measure space is the abstract global operator candidate. In the orthonormal-polynomial basis it becomes the infinite Jacobi operator. The same Jacobi operator acts as a generator when used in `exp(-itJ)`.

## Finite realization

The finite matrices are leading truncations of one coefficient sequence. Prefix consistency, positive beta coefficients, self-adjointness, and orthogonality are tested explicitly.
