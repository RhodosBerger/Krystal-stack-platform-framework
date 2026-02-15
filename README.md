# Dev-conditional: Heterogeneous Industrial AI Platform

This repository hosts a next-generation heterogeneous industrial AI platform, bridging the gap between high-level reasoning and real-time industrial control. It represents a fundamental rethink of how we architect compute at the edge, moving away from monolithic designs to a segmented, best-tool-for-the-job architecture.

---

## 1) Gamesa Cortex V2 (Primary Platform, 60%)
**The Neural Control Plane**

Gamesa Cortex V2 is a heterogeneous AI stack designed to run on commodity PC hardware while delivering safety-critical performance for industrial automation. It serves as an operating system for decision-making, orchestrating low-level acceleration through high-level logic.

### Core Architecture
- **Rust for Safety-Critical Planning** 🦀: Replaces Python in the critical path. Planning algorithms (A*, RRT) are compiled into shared libraries for zero-cost abstractions and memory safety.
- **Vulkan for Spatial Awareness** 🌋: Leverages Compute Shaders (Intel Iris Xe, NVIDIA RTX) for massive parallel voxel collision detection, treating the workspace as a live volumetric grid.
- **Economic Governance** ⚖️: A bio-inspired "Economic Governor" manages computation budgets. High-value tasks (Safety) get priority, while low-value tasks wait for "fiscal replenishment," preventing thermal throttling.
- **Docker & vGPU Framework** 🐳: A custom **vGPU Manager** creates "Virtual Slices" of the host GPU for containerized AI workloads, enabling deployment on any Linux distro.

**Codebase**: `gamesa_cortex_v2/`

---

## 2) FANUC RISE v3.0 - Cognitive Forge (Secondary Branch, 40%)
**Advanced CNC Copilot**

FANUC RISE v3.0 represents the evolution from deterministic execution to probabilistic creation. It is a **Conceptual Prototype & Pattern Library** demonstrating architectural patterns for bio-mimetic industrial automation.

### Key Concepts
- **Cognitive Forge**: Shifts focus from "Doing What Is Told" to "Suggesting What Is Possible," where AI proposes optimization strategies for operator selection.
- **Shadow Council Governance**: A multi-agent system (Creator, Auditor, Accountant) ensuring safe AI integration by validating probabilistic proposals against deterministic physics.
- **The Probability Canvas**: A "Glass Brain" interface visualizing potential futures and decision trees instead of just current status.
- **Neuro-Geometric Architecture**: Integer-only neural networks for edge computing.

**Codebase**: `advanced_cnc_copilot/`

---

## Repository Map

- **`gamesa_cortex_v2/`**: **Core Platform**. The active development branch for the heterogeneous AI stack (Rust/Vulkan/Python).
- **`advanced_cnc_copilot/`**: **Industrial Application Layer**. The FANUC RISE v3.0 Cognitive Forge prototype and pattern library.
- **`openvino_oneapi_system/`**: **Legacy/Foundation**. Previous generation performance orchestration layer (Krystal Vino). Served as the foundation for Cortex V2's optimization strategies.
- **`docs/`**: Additional technical materials and documentation.

---

## Direction Note

The priority of this repository is **Gamesa Cortex V2** as the main platform for PC hardware and inference performance. **FANUC RISE v3.0** serves as the advanced application layer and pattern library for industrial logic.

---

## Author & License

**Author**: Dušan Kopecký
**Email**: dusan.kopecky0101@gmail.com
**License**: Apache 2.0 (See `LICENSE` file)
