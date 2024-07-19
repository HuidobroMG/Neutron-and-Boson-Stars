# README.md
In this repository you will find Python codes and data files which are useful in the field of Neutron Stars (NS) physics. The codes can read or create Equations Of State (EOS) and solve the Tolman-Oppenheimer-Volkoff (TOV) system of differential equations in different cases. The data files which are already in the EoS_data folder of this repository contain the main state-of-the-art nuclear physics EOS and also some others obtained from the Skyrme model.

Units.py

BNStars.py

BNStars_Parallel.py

Tidal_BNStars.py

SlowRot_BNStars.py

Stars.py can solve the TOV system of equations for a given EOS data file. Furthermore, if the model has no crust you may add it at some value of the pressure, by default it will use the Barcelona-Catania-Paris-Madrid (BCPM) EOS.

SlowRotStars.py can solve an extended version of the system of differential equations from the Hartle-Thorne approximation. It calculates for a given EOS data file the first electric and magnetic multipolar deformations caused by tidal forces and for small rotations of the NS.

Generalized_Hybrid_EoS.py constructs the Generalized and Hybrid Skyrme model EOS defined in [https://arxiv.org/pdf/2006.07983.pdf] for different values of the transitions at high (core) and low (crust) densities.
