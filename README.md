# D-PLBA
D-PLBA is a dynamic PLBA prediction framework that jointly models the conformational evolution of pocket-ligand complexes and the corresponding changes in their interaction energies. Its core idea is to generate multiple dynamically evolving binding conformations from an initial protein-ligand complex and exploit both their structural and energetic information for subsequent affinity prediction.

<img src="./image/intro.png" alt="model"  width="70%"/>

## Installation
**Update**: Now the codes are compatible with PyTorch Geometric (PyG) >= 2.0.
### Dependency
The codes have been tested in the following environment:
Package  | Version
--- | ---
Python | 3.8.0
PyTorch | 1.8.0
CUDA | 11.7
PyTorch Geometric | **2.3.1**
RDKit | 2022.09.5
BioPython | 1.79
### Install via conda yaml file (cuda 11.7)

**The D-PLBA framework. (a) Prediction process of pocket-ligand motion trajectories and interaction energies. (b) PLBA prediction process.**
<img src="./image/method.png" alt="model"  width="70%"/>

**Four randomly selected samples showing predicted and reference pocket-ligand dynamic trajectories. The enlarged view of 6OA3 highlights the binding pocket region, illustrating trajectory evolution and corresponding interaction energies.**
<img src="./image/result.png" alt="model"  width="70%"/>

The PLBA dataset link: https://zenodo.org/records/22069317?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJjMDgwOGQwLTA0MTUtNGQ2Yi04MmMzLTkyNjYyNDhhM2NlNiIsImRhdGEiOnt9LCJyYW5kb20iOiIxZjIwNGVlNWFmNWQzYTYwYmM4MGEwYmM3ZGEzMWFmZCJ9.67cMf59E09gRCC6FrPBWSqVTTtEBrIvSNqpx-qtHPRpGjMzUuW5Z9z3N5jVC9ZFoNYwmCoLD_F7WyO0GuFpEbw

The trained model link: https://drive.google.com/file/d/1K18LaKkUnKy6Mdhcg7E0xTgVa7sHf2Rm/view?usp=drive_link
