### Introduction
A python project of data analysis for disk galaxy simulation snapshot files.


### Targets
1. Finish the basic analysis of a hdf4 snapshot file from galactic simulation.  
2. Plot the ordinary figures in simulation.
3. Complete it in a long study practice.


### <a id="contents">Contents</a>
1. <a href="#scheme">Physical analysis scheme</a>
2. <a href="#usage">Usage & Installation</a>
3. <a href="#code">Code structure</a>


<a id="scheme"></a>
### Design scheme <a href="#contents"><font size=4>(contents)</font></a>
#### Overview:
The `SnapPart` means snapshot partner, which treat every single snapshot as an object, and implement 
the attributes and methods of each snapshot. 

There are two main classes: 
- `single_snapshot_partner`: for analysis of file(s) of a single snapshot.
- `snapshots_partner`: based on `single_snapshot_partner`, for analysis of time sequence of many 
snapshots.

#### Analysis of a single snapshot: 
Pre process: recenter the system of a simulation, and align the disk into the $X-Y$ plane, then 
calculate the common coordinates of the particles.

Basic structural and dynamical quantities of the model: the bar major axis, number of arms, m2 symmetry 
amplitude, bar length ...

Dynamical quantities of particles: kinematical energy, angular momentum, circularity, orbital frequency,
guiding radius, actions ...

Physical quantification of the model or one mono-age-population: rotation curves, Toomre instability 
parameter, metallicity gradients, dispersion profiles, ...

Structure decomposition: decomposition of the different structural and kinematical/dynamical components 
in the model.

#### Analysis of time sequence of snapshots:
After necessary analysis of each snapshot, calculate the quantification of the model in a time sequence: 
the pattern speed, buckling strength, bar strength variation ...


<a id="usage"></a>
### Usage & Installation <a href="#contents"><font size=4>(contents)</font></a>


<a id="code"></a>
### Code structure <a href="#contents"><font size=4>(contents)</font></a> 
