# gpt

This repository contains a simple Python implementation of the steepest descent
method with exact line search for the quadratic function

\[
q(x) = \tfrac12 (10 x_1^2 - 18 x_1 x_2 + 10 x_2^2) + 4 x_1 - 15 x_2 + 13.
\]

Run the script to see convergence details from the four requested starting
points:

```bash
python steepest_descent.py
```

The script prints the theoretical linear convergence factor and an estimate of
the limsup of the ratio \((q^{k+1} - q^*) / (q^{k} - q^*)\) for each start.
