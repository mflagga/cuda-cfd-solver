# GPU-Accelerated CFD Solver

## Overview
A high-performance computational fluid dynamics (CFD) simulation engine developed in **C++** and **CUDA**. The project simulates incompressible flow using the Navier-Stokes equations (Stream function-vorticity formulation) and models mass transport via the Advection-Diffusion equation.

The solver leverages GPU parallelism (Red-Black Gauss-Seidel relaxation) to achieve high computational efficiency, while **Python** is used for visualization.

## Features
* **Navier-Stokes Solver:** Solves for the stream ($\psi$) and vorticity ($\zeta$) functions.
* **Mass Transport:** Simulates the advection and diffusion of a substance within the velocity field.
* **GPU Acceleration:** Custom CUDA kernels for finite difference calculations.
* **Visualization Pipeline:** Automated generation of velocity fields and transport animations using Matplotlib and FFmpeg.

## Animation
![Simulation Demo](images/transport.gif)

## Requirements
* NVIDIA GPU with CUDA Toolkit installed
* `nvcc` compiler
* Python 3.x (NumPy, Matplotlib)
* FFmpeg

## Usage
1.  **Build everything:**
    ```bash
    make
    ```
2.  **Complie and run the simulation to generate data:**
    ```bash
    nvcc main.cu -o fluid.out -arch=native -O3
    ./fluid.out
    ```
3.  **Generate visualizations:**
    ```bash
    make flow.png
    make transport.mp4
    ```
4.  **Delete data files, frames and executables**
    ```bash
    make clean
    ```

## Technologies
* C++ / CUDA
* Python (NumPy, Matplotlib)
* Make