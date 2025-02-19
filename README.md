# NEOPAX - Neoclassical Transport Package in JAX

NEOPAX aims to be a simple JAX framework to solve radial transport equations for stellarators and use it to optimise for different neoclassical quantities of interest.

## Quick Start
The main packages used by NEOPAX can be installed via pip:

```
pip install matplotlib
```
For GPU enable jax use this (it requires cuda12 installation, with the corresponding libraries, see JAX documentation for more details on this):
```
pip install -U "jax[cuda12]"
```
For CPU only usage use instead: 
```
pip install -U jax
```
The following packages are also needed:
```
pip install equinox
pip install lineax
pip install optimistix
pip install diffrax
```

At the moment the code makes use of a Monkes-JAX monoenergetic database Dij(r,collisionality,Er). To obtain these you can use Monkes-JAX, see: https://github.com/eduardolneto/monkes  

## Examples
To see examples of flux calculation or solving for the electric field simply go to the folder NEOPAX/tests and into one of the example folders you can run the respective script by 

'''
python script.py
'''

## Contributing
