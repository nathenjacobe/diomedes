# Diomedes: a GPU-based physics engine using Augmented Vertex Block Descent (AVBD)

https://github.com/user-attachments/assets/9f453d65-2f2c-4fb1-a0d5-b13cb887dff8 

(from left to right, the constraints are: hookean spring, double-pendulum, spherical, rope)

https://github.com/user-attachments/assets/491501fc-7908-44b7-9690-1a5a991481fa

Diomedes is a simple physics engine which uses the heavily parallelisable and numerically stable AVBD constraint solver (proposed by the University of Utah [here](https://graphics.cs.utah.edu/research/projects/avbd/)) on the GPU using [rust-gpu](https://rust-gpu.github.io/). It uses the [Vulkan](https://vulkan.org/) graphics API (using [ash](https://github.com/ash-rs/ash)).

Diomedes uses a broad phase followed by a narrow phase. Depending on the geometry pair, the narrow phase uses either GJK + EPA or SAT. Detected contacts are converted into constraints and resolved by the same AVBD solver used for the other constraints. Rope constraints currently enforce only a maximum distance; they have no elasticity and do not yet model bending.

Credits:
- AVBD formulation and reference implementation [here](https://github.com/savant117/avbd-demo3d)
- [Adrien Bennadji](https://github.com/adrien-ben) for [vulkan-tutorial-rs tutorial repo](https://github.com/adrien-ben/vulkan-tutorial-rs) which was used as a reference
