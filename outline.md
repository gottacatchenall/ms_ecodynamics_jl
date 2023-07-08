# Abstract

Ecosystems are often modeled using differential equations.

Platform for simulation of both determinsitic and stochastic models of
population and community dynamics, either at a single location or across a
spatial graph with environmental variation. 

Built on DifferentialEquations.jl, a state-of-the-art library for DEs. 

Enables inferences of dynamics. First Bayesian inference of DE parameters.


# Introduction 

Ecosystems are inherently dynamic---the abundance and composition of different species at
a different places is constantly changing over time. 


The study of the stability of dynamical systems over time has deep connections
with the modeling of ecosystem dynamics, e.g. the study of chaotic dynamics
emerged from @May1972's model of the logistic map undergoing a fork bifurcation
under increasing growth rates.


Models of the change of ecosystems over time (_ecosystem dynamics_) typically take
form of differential equations [@Lotka, @Volterra] (where time is continuous) or
difference-equations [@May] (where time is discrete). 

Species abundances inherently changes over time as a result of niche and neutral
processes, combined with dispersal and speciation [@Velland2010ConSyn]. 


Here, we present `EcoDynamics.jl`, a software toolkit for simulating a wide
variety of models of population and community dynamics, both in isolation and as
reaction-diffusion dynamics on spatial graphs. `EcoDynamics` enables both
deterministic and stochatic dynamics, and has the potential for environmental
covariates associated with nodes in the spatial graph to affect demographic
parameters, enabling virtual experiments about the effects of spatial environmental variation
on population and community processes.


 [@Thompson2020]

EcoDynamics.jl includes a rich library of models from the literature that enables easy addition
of customized models.

This toolkit is built upon DifferentialEquations.jl, a library for ordinary and
stochastic differential equations.

Why is this integration important?
- Orders of magnitude faster tha deSolver, for example. 
- Native integrations with other Julia packages for inference.
- Turing.jl for Bayesian inference of parameters 
- SciML, DiffEqFlux for PINNs

- Arbitrarily complicated models of stochasticity.

Development of Scientific Machine-Learning [SciML; @Rackauckas2021UniDif]. 
Combine mechanistic, domain specific models (in the form of DEs) with
data-driven models. 

Model a system as 

$$\frac{dx}{dx} = f(x, \theta) + g(x)$$

Where $f$ is a model of dynamics (e.g. Lotka-Volterra), and $g$ is a neural
network that, aftrer training, accounts for the difference in dynamics between a set of
observed datapoints $(x_i,t_i)$ and the hypothesized model $f$. 

Major angle: simulation of observing dynamics is impt. We can never observe
ground truth in nature. So, we never know if our statistical analyses we employ
would be capable of capturing an effect if it was real. We can attempt to
estimate this using power analysis, but simulatoin provides us a case where we
can have a 'virtual laboratory' [@VolkerGrimm] which can provide "ground-truth"
that we use. Quite common in physical sciences: test your analysis software on
simulated data to make sure it works _before_ you try using it with real data. 

# Software Design

## Brief software overview

- dispersal kernel
    - dispersal potential as a concept 

## Model index

Note that all `Population` and `Community` models can be run on spatial graphs.

| **Scale**      | **Name**                   | **Time**            | **Aliases**           | **Reference**  |
|----------------|----------------------------|---------------------|-----------------------|----------------|
| Population     | Logistic Model             | Continuous          | -                     |                |
| Population     | Beverton-Holt              | Discrete            | -                     |                |
| Population     | Ricker Model               | Discrete            | -                     |                |
| Population     | SIR Model                  | Discrete            | -                     |                |
| Community      | Trophic Lotka-Volterra     | Continuous          | -                     |                |
| Community      | Rosenzweig-Macarthur       | Continuous          | LV w/ Holling Type II |                |
| Community      | Holling Type III           | Continuous          |                       |                |
| Community      | DeAngelis-Beddington       | Continuous          |                       |                |
| Community      | Yodzis-Innes               | Continuous          |                       |                |
| Community      | Competitive Lotka-Volterra | Continuous          |                       |                |
| Metapopulation | Levins Metapopulation      | Continuous/Discrete |                       |                |
| Metapopulation | Incidence-Funciton Model   | Discrete            |                       | [Hanski1994]   |
| Metacommunity  | Neutral Model              | Continuous          |                       | [Hubbell2001]  |
| Metacommunity  | Process-based Competition  | Discrete            |                       | [Thompson2020] |

# Case studies 

Here we consider a few vignettes of how EcoDynamics.jl can be used. 

## Adding custom dynamics

Let's add a new model. SIR model comes be default. Let's add the SIRS version.


## Bayesian inference of local community dynamics 

- Local community dynamics (_community_)
    - Ease of integration with Turing for Bayesian inference of dynamics
    - It being built on DiffEq means easy integration w/ SciML. 
- Do a little power analysis on time frequency

## Mapping the transition between neutral and niche dynamics in competitive metacommunities

- Use case example: 
    - Competition across gradient, look at transition between neutral and niche
      or whatever

## Sources and sinks metapop

- Test Holt's claim about proprtion of sites that are sinks and when total
  extinction happens as function of how high the growth rate is in source patches.

## missing physics

- Use SciML to forecast Holling Type-II 
- Alternative to [BoettigerPaper on forecasting]
- Ask Victor B for help?

# Structural identifiability?

# Discussion

Comparing different simulated models of dynamics to empirical data is useful.
- This enables stiatistical comparison of potentil models for producing a given sampled dataset. 

- Deep learning for forecasting. MCMC estimate of parameters given true model is
  ideal [BoettigerPaper]
- SciML and forecasting. Fill in "missing physics/ecology" to determine what the
  _true_ dynamics are

