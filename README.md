ocl-mod-eqn-solver
==================

This solver uses OpenCL and iterative hill climbing to attemp to solve as many equations as possible. Each work-item starts at a random location and all threads hill climb in parallel.


Only `solver.cl` and possibly `ocl-solver.cpp` are of any interest. The rest is just support framework/test.
