# Summary of the MGS Build Environment

Here's a summary of what we've learned about building the Model Graph Simulator (MGS) framework:

## System Requirements and Dependencies

1. **Core Dependencies**:
   - Bison ≥ 2.4.1 (installed via Homebrew, now at 3.8.2)
   - Flex ≥ 2.5.4 (Apple flex-35 version 2.6.4)
   - m4 ≥ 1.4.17 (installed via Homebrew, now at 1.4.19)
   - GMP library (installed via Homebrew)
   - MPI implementation (Open MPI installed via Homebrew)
   - Python with pybind11 (Python 3.10.16 with pybind11 2.13.6)
   - SuiteSparse/CXSparse library (installed via Homebrew)
   - MNIST header files (downloaded to gsl/utils/mnist-master)

2. **Environment Variables**:
   - NTSMGSROOT, NTSROOT, MGSROOT: Set to the project root directory
   - GSLROOT: Set to $NTSROOT/gsl
   - MDLROOT: Set to $NTSROOT/mdl
   - PATH: Updated to include $NTSROOT/gsl/bin
   - GSLPARSER: Set to $NTSROOT/gsl/bin/gslparser
   - SUITESPARSE: Set to Homebrew's suite-sparse installation
   - PYTHON_INCLUDE_DIR: Set to Python include directory

## Build System Structure

1. **Main Components**:
   - MDL (Model Definition Language) parser
   - Common library
   - GSL (Graph Specification Language) parser
   - NTI (Neural Tissue Simulator) library

2. **Build Scripts**:
   - `build_script`: Main build coordinator for all components
   - `build_gsl`: Specific script for building the GSL component
   - `set_env`: Sets up environment variables

3. **Build Modes**:
   - Debug: Default build mode with debugging symbols
   - Release: Optimized build with `-release` flag

4. **Platform Support**:
   - Originally designed for Linux
   - Modifications needed for macOS compatibility
   - Container-based build available via Docker

## Build Workflow

1. **Preparation**:
   - Install all dependencies
   - Source the environment setup: `source set_env`

2. **Building**:
   - Use modified `build_script` with macOS platform: `./build_script -p MACOS -j 4 --rebuild --release`
   - Alternative: Build components individually

3. **Key Script Modifications for macOS**:
   - CPU detection using `sysctl` instead of `/proc/cpuinfo`
   - Library detection using Homebrew paths instead of `ldconfig`
   - Flex version parsing adjustment for Apple's custom version format
   - Platform-specific checks and implementations

This summary provides a foundation for understanding how to build the MGS framework on macOS, the dependencies required, and the structure of the build system.