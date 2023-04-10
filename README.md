# Projet_IA

## Important Information
* Requires CARLA to function 
* When closing close the client first!!!
* **AFTER** you can close the server if you please
* Its not recommended to use the radar sensor as it is incomplete
* [Carla docs](https://carla.readthedocs.io/en/latest/)
* [Carla Quickstart](https://carla.readthedocs.io/en/latest/start_quickstart/)

## Python Dependencies
* numpy
* pygame
* carla
* scikit-image

## Application dependencies
* [Carla](https://github.com/carla-simulator/carla/archive/refs/tags/0.9.14.zip)

## Instructions
Requires Python 3.8 or older 

### Important commands
* python path/to/projet.py 
  * -x is for the path to the track file
  * -s is for the path to the server file (DO NOT USE if a server instance already active)

## Troubleshooting 
* If the server is closed before the client make sure to force shutdown carla background processes in task manager