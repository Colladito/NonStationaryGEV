# Documentation for Automatic Time-Dependent GEV fit: GEVnonstat_v2

The aim of this class is to select and calculate the parameters which minimize the AIC related to time-dependent GEV distribution using the Maximum Likelihood method within an iterative scheme, including one parameter at a time based on a perturbation criteria. The process is repeated until no further improvement in the objective function is achieved. 

The main function of this class is the AutoAdjust.

## Class Definition

## Input variables
- `xt`: Environmental maxima data.
- `t`: Time when maxima occurs in yearly scale.
- `kt`: Frequency parameter to measure the importance of the number of points.
- `covariates`: covariate dictionary where each column corresponds to the values of a covariate at the times when maxima occurs (Dictionary containing the data).
- `trends`: Number of trends to introduced
- `example`: Boolean to see if a plot has to be created.
- `quanval`: Quantile value to be plotted (default 0.95).
- - harm: Boolean if harmonic part should be introduced

### Input Variables type:

| Variable     | Type          | Default       |
|--------------|---------------|---------------|
| `xt`         | numpy.array   |       -       |
| `t`          | numpy.array   |       -       |
| `kt`         | numpy.array   |      None     |
| `covariates` | dictionary    |      None     |
| `trends`     | Boolean       |      False    |
| `example`    | String        |      None     |
| `quanval`    | 0-1 float     |      0.95     |

## Functions
Functions implemented in the `GEVnonstat2` class.

- `__init__`: Initialization of the data.
- `validate_data`: Validate the data introduced.
- `AutoAdjust`: Perform the algorithm implemented (Most important function). Inputs: 
    - `maxiter`: number of iterations for the algorithm (100 as default).
    - `plot`: boolean value which calls the plot function (False as default).
- `_AIC`: Compute the Akaike Information Criterion given certain loglikelihood value and the number of parameters.
- `_optimize_parameters`: Function to estimate the parameters of the Time-Dependent GEV distribution using the Maximum Likelihood Method.
- `_auxmin_loglikelihood`: Auxiliar function with the loglikelihood function to use in the `_optimize_parameters` minimization step.
- `_auxmin_loglikelihood_grad`: Auxiliar function with the gradient of the loglikelihood function to use in the `_optimize_parameters` minimization step.
- `compute_numerical_hessian`: Auxiliar function to compute the numerical hessian matrix of the loglikelihood.
- `_loglikelihood`: Function which calculate the loglikelihood, the Gradient and Hessian given certain parameters.
- `_parametro`: Calculates the value of the location, scale and shape parameter, one in each call. 
- `_search`: Auxiliar function used in `_parametro` to search the nearest value of certain time.
- `_evaluate_params`:Function to evaluate the parameters in the corresponding values ($\beta_0,\beta,\dots$).
- `_Dparam`: Derivative of the location, scale and shape functions with respect to harmonic parameters. It corresponds to the rhs in equation (A.11) of the paper.
- `_quantile`: Calculates the quantile q associated with a given parameterization, the main input is quanval introduced in __init__ (default 0.95).
- `plot`: Plot the location, scale and shape parameter and also the PP-plot and QQ-plot.
- `QQplot`: QQ plot
- `_Zstandardt`: Calculates the standardized variable corresponding to the given parameters.
- `_Dzweibull`: Calculates the derivatives of the standardized maximum with respect to parameters
- `_Dmupsiepst`: Calculates the derivatives of the standardized maximum with respect to parameters
- `_DQuantile`: Calculates the quantile derivative associated with a given parameterization with respect model parameters
- `PPplot`: PP plot
- `_CDFGEVt`: Calculates the GEV distribution function corresponding to the given parameters.
- `ReturnPeriodPlot`: Function to plot the Aggregated Return period plot for each month and if annualplot, the annual Return period (default True)
- `_aggquantile`: Function to compute the aggregated quantile for certain parameters
- `_fzeroquanint`: Function to solve the quantile.
- `_fzeroderiquanint`: Function to solve the quantile
- `_ConfidInterQuanAggregate`: Auxiliar function to compute the std for the aggregated quantiles


# TO DO
## Cosas que hacer:
- Si añadimos las covariables en el parametro de forma pero el resultado optimo anterior es gumbel, la introducciond de las covaribles $\gamma_i^{co}=0$ provocan que $\xi_t=0 \forall t$ entonces cómo debería introducir estos parámetros?

- Añadir los multiplicadores de Lagrange (linea 715) HECHO 

- Definir bien la funcion self.plot(), y añadir las funciones auxiliares necesarias para los plots HECHO

- Ver los vlaores negativos en invI0 con valores negativos en la diagonal. Deberian ser todos positivos HECHO

- Completar la documentacion de las funciones y demas

- Probar con los nuevos datos

## Versions:
- v1: Normal version (solver trust-constr)
- v2: Advanced version changing some style of the plots (solver trust-constr)
- v3: Changing the optimization solver to reduce time


### Optimization methods which use the Hessian:

Newton-CG, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

It is better to use 'trust-constr' since it returns the lagrangian values as it is obtained in Matlab.

## Solved Problems:
- Interaccion de `varphi`/`varphi2` en el proceso iterativo de las covariables fallaba a la hora de añadir nuevas Cambios:
    - Linea 303: `varphiini_loc = np.append(auxvarphi_loc[list_loc], [0])` /// Previamente: `varphiini_loc = np.append(varphi, [0])`
    - Linea 309: `varphiini_sc = np.append(auxvarphi_sc[list_sc], [0])` /// Previamente: `varphiini_sc = np.append(varphi2, [0])`

# Analisys of different solvers

### Matlab Data
| Parameter            | L-BFGS-B |   TNC   | SLSQP | trust-constr |
|----------------------|----------|---------|-------|--------------|
| $\beta_0$            |  5.135   | 5.1333  |       |   5.135      |
| $\beta_1$            |   1.917  | 1.9484  |       |   1.917      |
| $\beta_2$            |   0.131  | 0.1628  |       |    0.131     |
| $\beta_3$            |   -0.188 |   --    |       |    -0.188    |
| $\beta_4$            |  -0.0694 |   --    |       |   -0.0694    |
| $\beta_5$            |  -0.132  |   --    |       |   -0.132     |
| $\beta_6$            | 0.04885  |   --    |       |   0.04884    |
| $\beta_7$            | 0.13786  |   --    |       |    0.13788   |
| $\beta_8$            | -0.0858  |   --    |       |   -0.08586   |
| $\alpha_0$           |  0.139   | 0.1583  |       |   0.139      |
| $\alpha_1$           |  0.4097  | 0.4038  |       |   0.4097     |
| $\alpha_2$           |  0.066   | 0.0629  |       |  0.066       |
| $\alpha_3$           |  -0.118  | -0.1027 |       |   -0.118     |
| $\alpha_4$           |  0.0291  | 0.0531  |       |    0.029     |
| $\alpha_5$           |    --    |   --    |       |    --        |
| $\alpha_6$           |    --    |   --    |       |    --        |
| $\alpha_7$           |    --    |   --    |       |    --        |
| $\alpha_8$           |    --    |   --    |       |    --        |
| $\gamma_0$           |    --    |   --    |       |    --        |
| $\gamma_1$           |    --    |   --    |       |    --        |
| $\gamma_2$           |    --    |   --    |       |    --        |
| $\beta_{T}$          |    --    |   --    |       |    --        |
| $\beta_{1}^{co}$     |    --    |   --    |       |    --        |
| $\beta_{2}^{co}$     |    --    |   --    |       |    --        |
| $\beta_{5}^{co}$     |    --    |   --    |       |    --        |
| $\beta_{6}^{co}$     |    --    |   --    |       |    --        |
| $\beta_{7}^{co}$     |    --    |   --    |       |    --        |
| $\alpha^{T}$         |    --    |   --    |       |    --        |
| $\alpha_{1}^{co}$    |    --    |   --    |       |    --        |
| $\alpha_{2}^{co}$    |    --    |   --    |       |    --        |
| $\alpha_{5}^{co}$    |    --    |   --    |       |    --        |
| $\alpha_{6}^{co}$    |    --    |   --    |       |    --        |
| $\alpha_{7}^{co}$    |    --    |   --    |       |    --        |
| $\alpha_{8}$         |    --    |   --    |       |    --        |
| AIC                  | 2508.664 | 2517.693|       |   2508.664   |
| Time                 |   8.0s   |  8.6s   |       |   1m 44.1s   |

### Bretagne Data
| Parameter            | L-BFGS-B |   TNC   | SLSQP | trust-constr |
|----------------------|----------|---------|-------|--------------|
| $\beta_0$            |  5.6117  | 5.7084  |       |    5.6817    |
| $\beta_1$            |  2.5248  | 2.3952  |       |    2.5274    |
| $\beta_2$            |  0.3689  | 0.3483  |       |    0.3832    |
| $\beta_3$            |    --    |         |       |      --      |
| $\beta_4$            |    --    |         |       |      --      |
| $\beta_5$            |    --    |         |       |      --      |
| $\alpha_0$           |  0.1761  | 0.4018  |       |    0.3565    |
| $\alpha_1$           |  0.3523  |         |       |    0.3655    |
| $\alpha_2$           |  0.0527  |         |       |    0.0522    |
| $\alpha_3$           | -0.1093  |         |       |   -0.1153    |
| $\alpha_4$           |  0.0273  |         |       |    0.0187    |
| $\alpha_5$           |    --    |         |       |      --      |
| $\alpha_6$           |    --    |         |       |      --      |
| $\alpha_7$           |    --    |         |       |      --      |
| $\alpha_8$           |    --    |         |       |      --      |
| $\gamma_0$           |    --    |         |       |      --      |
| $\gamma_1$           |    --    |         |       |   -0.0213    |
| $\gamma_2$           |    --    |         |       |   -0.0159    |
| $\beta_{T}$          |  0.0150  | 0.0112  |       |    0.0119    |
| $\beta_{1}^{co}$     |  0.6245  | 0.6977  |       |    0.6218    |
| $\beta_{2}^{co}$     | -0.5049  | -0.5947 |       |   -0.5176    |
| $\beta_{4}^{co}$     | -0.1176  | -0.1560 |       |   -0.1199    |
| $\beta_{7}^{co}$     | -0.1599  | -0.1719 |       |   -0.1603    |
| $\beta_{8}^{co}$     | -0.1343  | -0.1334 |       |   -0.1198    |
| $\alpha^{T}$         |          | -0.0082 |       |   -0.0084    |
| $\alpha_{0}^{co}$    | -0.0429  | -0.0314 |       |   -0.0364    |
| $\alpha_{1}^{co}$    |  0.1246  |  0.1241 |       |    0.1315    |
| $\alpha_{2}^{co}$    |    --    | -0.0387 |       |       --     |
| $\alpha_{5}^{co}$    | -0.0910  | -0.1053 |       |   -0.1191    |
| $\alpha_{6}^{co}$    |  0.0861  |  0.0478 |       |    0.0516    |
| AIC                  | 1910.374 | 1941.697|       |   1904.367   |
| Time                 |  11.3s   |   7.0s  |       |   1m 19.3s   |



# Preguntas
- Al añadir el parametro de tendencia en el parametro de escala, si el tiempo que se tiene es muy grande (mas de 40 años) entonces genera valores de $\psi_t$ que tienden a cero provocando que el valor $x_n$ tienda a infinito para tiempos altos.
- Añadir cota al parametro de tendencia? No tiene sentido que la tendencia sea un valor grande, linea 702
