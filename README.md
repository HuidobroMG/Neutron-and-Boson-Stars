# Neutron-Stars
In this repository you will find Python codes and data files which are useful in the field of Neutron Stars (NS) physics. The codes can read or create Equations Of State (EOS) and solve the Tolman-Oppenheimer-Volkoff (TOV) system of differential equations in differente cases. The data files which are already in the repository contain the main state-of-the-art nuclear physics EOS and also some others obtained from the Skyrme model.

The codes are explained below:

Stars.py can solve the TOV system of equations for a given EOS data file. Furthermore, if the model has no crust you may add it at some value of the pressure, by default it will use the Barcelona-Catania-Paris-Madrid (BCPM) EOS.

SlowRotStars.py can solve an extended version of the system of differential equations from the Hartle-Thorne approximation. It calculates for a given EOS data file the first electric and magnetic multipolar deformations caused by tidal forces and for small rotations of the NS.

The EoS_Data folder contains 5 EOS obtained from the Skyrme model and the BCPM EOS.
