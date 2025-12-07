# Building MGS

Detailed build instructions for the Model Graph Simulator framework.

## Quick Start
```bash
# Build for Linux (default target)
./build_mgs -p LINUX --as-MGS

# Or build with GPU support
./build_mgs -p LINUX --as-GPU
```

## Build Scripts

### Main Build Script
```bash
./build_script  # Print help instructions
```

### GSL Build
```bash
./build_gsl -h                        # Print help
./build_gsl --rebuild --release -d 4  # Release build (recommended)
./build_gsl --rebuild -d 4            # Debug build
```

### NTI Build
```bash
make debug=yes    # Debug build
make              # Release build
```

## External Dependencies

### Required Libraries

- **bison** v2.4.1 or above (built using m4 >= 1.4.17)
- **flex** v2.5.4 or above
- **libgmp** (GNU Multiple Precision library)
- **python** with pybind11
  - Set `PYTHON_INCLUDE_DIR` environment variable
- **cxsparse** library
  - Set `SUITESPARSE` environment variable
- **MNIST headers** (for MNIST example)
```bash
  git clone --depth=1 https://github.com/wichtounet/mnist.git gsl/utils/mnist
  rm -rf gsl/utils/mnist/.git
```

### Installing Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install build-essential bison flex libgmp-dev python3 python3-dev
```

**macOS:**
```bash
brew install bison flex gmp python3
```

## Container-Based Build (Docker)

For reproducible builds and GPU support, use the Docker-based build system.

### Prerequisites

Ensure the proper NVIDIA driver is installed on the host machine.

- [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
- [CUDA 10.0 Downloads](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal)

### Step 1: Increase inotify Limit (if needed)

Only run this if you get an error when running the container:
```bash
sudo -i
echo 1048576 > /proc/sys/fs/inotify/max_user_watches
exit
```

### Step 2: Install nvidia-docker (Docker < 19.03)

Skip if `docker --version >= 19.03`
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Step 3a: Build the Docker Image

**On x86-64 machine:**
```bash
docker build --target devel-base -t mgs_baseimage -f Dockerfile.build .
```

**On ppc64le (Power8/Power9):**
```bash
docker build --target devel-base -t mgs_baseimage -f Dockerfile.build_ppc64le .
```

### Step 3b: Launch the Container

**With GPU support (Docker 19.03+):**
```bash
docker run --gpus all -it --name=mgs_dev \
  --mount src="$(pwd)",target=/home/mgs,type=bind \
  -e LOCAL_USER_ID=`id -u $USER` \
  mgs_baseimage /bin/bash
```

**With debug support (gdb):**
```bash
docker run --privileged --gpus all -it --name=mgs_dev \
  --mount src="$(pwd)",target=/home/mgs,type=bind \
  -e LOCAL_USER_ID=`id -u $USER` \
  mgs_baseimage /bin/bash
```

**Without GPU:**
```bash
docker run -it --name=mgs_dev \
  --mount src="$(pwd)",target=/home/mgs,type=bind \
  -e LOCAL_USER_ID=`id -u $USER` \
  mgs_baseimage /bin/bash
```

### Step 4: Build Inside Container
```bash
# You're now inside the container at /home/mgs
./build_script -p LINUX --as-GPU
# or
./build_script -p LINUX --as-GPU --release
```

**Note:** When building with `--as-GPU`, the system uses `./models/gpu.mdf` file.

**Note:** The container doesn't include editors (vi, emacs) to keep size small. Do code development on the host, then switch to container for compilation:
- Host → Container: `docker attach <CONTAINER-ID>`
- Container → Host: `Ctrl-p-q`

## Docker Management

### Common Docker Commands
```bash
# List images
docker image ls

# List running containers
docker container ls

# List all containers (including stopped)
docker ps -a

# Remove a container
docker rm <container-name>

# Return to a running container (use first 3+ digits of ID)
docker attach <CONTAINER-ID>
```

## Runtime Flags
```bash
gslparser -t <threads> -f <file.gsl> -s <seed>
```

- `-t` Number of threads
- `-f` GSL file to run
- `-s` Random number generator seed

## Troubleshooting

### Rebuild Issues

If the build seems to restart from the beginning:

The build needs to be successful at least once to enable continuous building. Start with minimal systems and add models incrementally.

### GPU Build Issues

1. Verify NVIDIA driver version matches CUDA requirements
2. Check that `--gpus all` flag is supported (Docker 19.03+)
3. Ensure nvidia-docker toolkit is installed

## Next Steps

After building:
- See [README.md](README.md) for architecture overview
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow
- See [examples/](examples/) for example simulations
