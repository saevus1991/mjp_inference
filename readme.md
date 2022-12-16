A package for simulation, integration and likelihood-based inference of structured Markov jump processes with an interface for building models. 

## Introduction

This package provides provides a model class that allows modular construction of a wide variety of structured Markov jump processes. The model can be used for Gillespie simulation, integration of the master equation, filtering and smoothing with discrete observations likelihood-based inference. The backend is written in C++ using the Eigen library for linear algebra operations. To simplify the use, these header-only libraries have been included in the repository. Interfaces are exposed to Python using Pybind11.

The package includes an interface to construct custom models with general hazard functions. These functions are JIT-compiled via Numba and passed to the C++ backend. 

## Installation

The code has to be compiled from source. For this, a compiler is required supporting C++14. First, setup a Python environment. Then, open a command prompt, activate the environment and navigate to the project directory. Now install the requirements

```
pip install -r requirements.txt
```

Now gow to the source directory and run the setup
```
cd mjp_inference/_c
python setup.py build
```

After completion of the build, you should be able to run the test files in `./tests/_c`. If required, add the top level of the project to your python path. 


## Usage

### Setting up a model

Models are constructed from a collection of species or agents and a number of events. As an example, consider a simple gene expression model consisting of species `G0`, `G1` representing active and inactive states of a gene and species `mRNA` and `Protein`. This is set up as 
```
import mjp_inference as mjpi

# set up model
model = mjpi.MJP("Simple gene expression model")
# add species
model.add_species(name='G0', default_value=1)
model.add_species(name='G1')
model.add_species(name='mRNA', upper=100)
model.add_species(name='Protein', upper=500)
```

The dynamics of a Markov jump process are governed by a number of events. Each event $i$ is associated with a hazard function $h_i(x)$ and a change vector $v_i$. The hazard function defines the probability for an event to happen in a small intervall $\tau$ as

$$
\mathrm{Pr}(X(t+\tau) = x + v_i \mid X(t) = x) = \tau h_i(x) + o(\tau)  .
$$

Sructured multi-component systems are typically sparse in the sense that $h_i$ will only depend on a subset of $x$ and $v_i$ only changes a subset of $x$. We call the subset on which $h_i$ depends the input species of the event. Similarly, the species that are changed by the event are called the output species. As an example, consider the trascription event in gene expression. In a simple model of gene expression, transcription occurs with a constant rate $c_\mathrm{tc}$ whenever the gene is in an active state. We can convert this into a function as $h_\mathrm{tc}(x) = c_\mathrm{tc} x_2$ where $x_2$ is the number of active genes $\mathrm G_1$. The set of input species is therefore $\\{ \mathrm G_1 \\}$. When the translation event occurs, a single $\mathrm{mRNA}$ is created. Thus, the set of output species is $\\{ \mathrm{mRNA} \\}$ and the change vector reduced to the output species is $(1)$. To add such an event to the model, use

```
model.add_event(mjpi.Event(name='Transcription', iput_species=['G1'], 
  output_species=['mRNA'], hazard=lambda x: c_tc * x[0], change_vec=[1]))
```

As another example, consider the activation of the gene that occurs with rate $c_\mathrm{on}$ whenever $\mathrm G_0$ is one and $\mathrm G_1$ is zero. This implies a hazard of the form $h_\mathrm{on} (x) = c_\mathrm{on} x_1$ with input species $\\{ \mathrm G_0 \\}$, output species $\\{ \mathrm G_0, \mathrm G_1 \\}$ and change vector $(-1, 1)$. 

Often, events are given in the form of a chemical reaction. The full system of events for the simple gene expression  model is

$$\begin{aligned}
&\text{Activation:} &\quad 1 \\, \mathrm G_0  &\longrightarrow 1 \\, \mathrm G_1 \\
&\text{Dctivation:} &\quad 1 \\, \mathrm G_1 &\longrightarrow  1 \\, \mathrm G_1\\
&\text{Transcription:} &\quad  1 \\, \mathrm{G_1} &\longrightarrow 1 \\, \mathrm G_1 + 1 \\, \mathrm{mRNA} \\
&\text{mRNA degradation:} &\quad 1 \\, \mathrm{mRNA} &\longrightarrow \emptyset \\
&\text{Translation:} &\quad 1 \\, \mathrm{mRNA} &\longrightarrow 1 \\, \mathrm{mRNA} + 1 \\, \mathrm{Protein} \\
&\text{Protein degradation:} &\quad 1 \\, \mathrm{Protein} &\longrightarrow \emptyset 
\end{aligned}$$

The species on the left side are understood as consumed by the event, while the species appearing on the right are understood as produced by the event. To convert this to the above formalism, input species are simply the species on the left. However, output species are only those that undergo a net change by the event. Thus, $\mathrm{mRNA}$ is not an output species of translation even though it appears on the right side. Manual conversion of the reaction formalism is not necessary. Instead, events can be added in the form 

```
model.add_event(mjpi.Reaction(name='Translation', reaction='1 mRNA -> 1 mRNA + 1 Protein',
  rate=0.01, propensity=lambda x: x[0]))
```

A common form of propensity function is mass-action dynamics. Here, the hazard is proportional to the combinatorial probability of all the involved particles meeting randomly. In this case, the hazard can be constructed automatically and it is enough to specify

```
model.add_event(mjpi.MassAction(name='Translation', reaction='1 mRNA -> 1 mRNA + 1 Protein',
  rate=0.01))
```

Therefore the full model is specified by

```
# add events
model.add_event(mjpi.MassAction(name='Activation', reaction='1 G0 -> 1 G1', rate=0.001))
model.add_event(mjpi.MassAction(name='Deactivation', reaction='1 G1 -> 1 G0', rate=0.001))
model.add_event(mjpi.MassAction(name='Transcription', reaction='1 G1 -> 1 G1 + 1 mRNA', rate=0.06))
model.add_event(mjpi.MassAction(name='mRNA Decay', reaction='1 mRNA -> 0 mRNA', rate=0.001))
model.add_event(mjpi.MassAction(name='Translation', reaction='1 mRNA -> 1 mRNA + 1 Protein', rate=0.01))
model.add_event(mjpi.MassAction(name='Protein Decay', reaction='1 Protein -> 0 Protein', rate=0.0009))
model.build()
```

The different ways to initiate an event have in common that they only require local quantities and no global ordering of species as typically implied by hazard functions of the form $h_i(x)$ . This ordering and the global hazard are constructed in `model.build()` with vector $x=(x_1, \ldots, x_n)$ corresponding to species in the order they were added to the model. The model can then be passed to `cc.Simulator` to simulate stochastic dynamics by the Gillespie algorithm or to `cc.MasterEquation` to solve the master equation of the system. For more possible applications, you an have a look at the files in `./tests/_c`.
