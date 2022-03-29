# Combining model-based and model-free replay in a changing environment
Here we analyse the benefits of combining Simulation Reactivations (SR) and Memory Reactivations (MR) in a robot
control architecture which includes both Model-based (MB) and Model-Free (MF) Reinforcement Learning (RL).
We thus investigate the effects of including replay in the algorithm proposed in
[Dromnelle(2020)](https://link.springer.com/chapter/10.1007/978-3-030-64313-3_8), which coordinates a
Model-based and a Model-free RL experts within the decision layer of a robot control architecture. Interestingly, this
algorithm had been previously tested in a navigation environment that includes open areas, corridors, dead-ends,
a non-stationary task with changes in reward location, and a stochastic transition function between states of the task.
In these conditions, previous results showed that the combination of MB and MF RL enables to (1) adapt faster to task
changes thanks to the MB expert, and to (2) avoid the high computational cost of planning when the MF expert has been
sufficiently trained by observation of MB decisions
([Dromnelle(2020)](https://link.springer.com/chapter/10.1007/978-3-030-64313-3_8)). Nevertheless, replay processes have not been
included in this architecture yet, and this is what has been explored with the presented code.
  
> This code goes with the following submission: Massi et al. (2022) Model-based and model-free
> replay mechanisms for reinforcement learning in neurorobotics. Submitted.

## Contributors
- Rémi Dromnelle
- [Jeanne Barthélemy](https://github.com/Gaerdil)
- Julien Canitrot
- [Elisa Massi](https://github.com/elimas9) 

## Usage
- *ENVIRONMENT_MAPS* contains all the file needed to similate the experimental environemnt and to plot it (more
  information in the *readme* inside the folder)
- *EXPERIMENT_PLOTS* contains the script used to plot the results contained in *EXPERIMENT_SIMULATION/logs* (more
  information in the *readme* inside the folder)
- *EXPERIMENT_SIMULATION* contains all the code for simulating an experiment (more
  information in the *readme* inside the folder)

## Questions?
Contact Elisa Massi (lastname (at) isir (dot) upmc (dot) fr)