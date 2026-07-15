# Reported numerical milestones

## Finite reconstruction

At depth 24, a high-precision validation run reported full 12-zero coverage and relative RMSE `1.707861e-4`, with prediction drift across stable precision/radius settings of about `1.25e-12` and direct-derivative/Cauchy moment agreement near `1.09e-88`.

## Stable-bulk GUE frontier

A depth-52 frontier scan reported a robust unchanged-spectrum block with 22 spacings, GUE preferred under all three unfolding methods, RvM-row KS values approximately `0.1323 / 0.1999 / 0.4131` for GUE/GOE/Poisson, mean ratio `0.6197`, and bootstrap GUE-win fraction `0.995`.

## Preregistered holdout

A frozen depth-64 protocol reported 28 stable spacings, GUE and ratio criteria passing under all three unfolding methods, minimum bootstrap GUE-win fraction `0.990`, and zero relative RMSE `3.633e-12`.

These are finite computational observations. Full output files from the user's machine are not all included here; reproduce them with the provided commands.
