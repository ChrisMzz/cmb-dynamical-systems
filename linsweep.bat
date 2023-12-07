@echo off
title Automated parameter sweeps
color 05

(
for %%p in (pi0, pi1, pi2, alpha1, alpha2, alpha3, alpha2b, alpha3b, beta1, beta2, gamma2, gamma3, delta0, delta2, tauC, tauA2, tauCA1, tauCA2) do (
    python sweep.py %%p linear save
    python sweep.py %%p quadratic save
    )
) > sweeplogs.txt

echo All done!
pause
