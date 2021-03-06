**7.9.** At http://www.statsci.org/data/general/brunhild.html, you will find a dataset
that measures the concentration of a sulfate in the blood of a baboon named
Brunhilda as a function of time. Build a linear regression of the log of the
concentration against the log of time.

**(a)** Prepare a plot showing (a) the data points and (b) the regression line in
log-log coordinates.

**(b)** Prepare a plot showing (a) the data points and (b) the regression curve
in the original coordinates.

**(c)** Plot the residual against the fitted values in log-log and in original coordinates.

**(d)** Use your plots to explain whether your regression is good or bad and why.

**7.10.** At http://www.statsci.org/data/oz/physical.html, you will find a dataset of measurements
by M. Larner, made in 1996. These measurements include body
mass, and various diameters. Build a linear regression of predicting the body
mass from these diameters.

**(a)** Plot the residual against the fitted values for your regression.

**(b)** Now regress the cube root of mass against these diameters. Plot the
residual against the fitted values in both these cube root coordinates and
in the original coordinates.

**(c) Use your plots to explain which regression is better.**

**7.11.** At https://archive.ics.uci.edu/ml/datasets/Abalone, you will find a dataset of
measurements by W. J. Nash, T. L. Sellers, S. R. Talbot, A. J. Cawthorn and
W. B. Ford, made in 1992. These are a variety of measurements of blacklip
abalone (Haliotis rubra; delicious by repute) of various ages and genders.

**(a)** Build a linear regression predicting the age from the measurements, ignoring
gender. Plot the residual against the fitted values.

**(b)** Build a linear regression predicting the age from the measurements, including
gender. There are three levels for gender; I’m not sure whether
this has to do with abalone biology or difficulty in determining gender.
You can represent gender numerically by choosing 1 for one level, 0 for
another, and -1 for the third. Plot the residual against the fitted values.

**(c)** Now build a linear regression predicting the log of age from the measurements,
ignoring gender. Plot the residual against the fitted values.

**(d)** Now build a linear regression predicting the log age from the measurements,
including gender, represented as above. Plot the residual against
the fitted values.

**(e)** It turns out that determining the age of an abalone is possible, but difficult
(you section the shell, and count rings). Use your plots to explain which
regression you would use to replace this procedure, and why.

**(f)** Can you improve these regressions by using a regularizer? Use glmnet to
obtain plots of the cross-validated prediction error.
