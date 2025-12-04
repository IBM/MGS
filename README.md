# MGS: Model Graph Simulator

**A high-performance framework for composing and executing large-scale heterogeneous network simulations**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS-lightgrey.svg)]()

## Overview

MGS (Model Graph Simulator) is a domain-specific language and execution framework designed at IBM Research for building complex network simulations from composable, interoperable models. Originally developed for massively parallel systems like IBM Blue Gene, MGS enables researchers to:

- **Define heterogeneous model types** that communicate through well-defined interfaces
- **Compose models into hypergraphs** where vertices are model instances and edges represent data dependencies
- **Execute efficiently** on distributed systems, multi-core CPUs, and GPUs
- **Scale simulations** from laptops to supercomputers without code changes

### Key Innovations

🔷 **Declarative Graph Specification**: Define network topology and model composition using intuitive domain-specific languages (MDL/GSL)

🔷 **Heterogeneous Model Integration**: Different model types coexist and interact within a single simulation

🔷 **Automatic Parallelization**: Framework handles distribution, communication, and synchronization

🔷 **Interface-Based Composition**: Models couple through typed interfaces, ensuring composability and reusability

## Architecture

MGS provides two complementary languages:

### Model Definition Language (MDL)
Defines computational model types with:
- Encapsulated state variables
- Communication interfaces (inputs/outputs)
- Execution phases (initialization, update, etc.)
- Connection predicates for selective coupling

### Graph Specification Language (GSL)  
Instantiates and organizes models into networks:
- Spatial grids and layers
- Connection patterns (functors)
- Hierarchical composites
- Phase scheduling

Together, these languages enable you to specify **what** to compute (model behavior) and **where/when** to compute it (graph structure and execution).

## Why MGS?

### For Graph Theorists & Network Scientists
- Native hypergraph representation
- Flexible topology specification
- Dynamic connection patterns
- Scales to millions of nodes

### For HPC Researchers
- MPI-based distributed execution
- GPU acceleration support (CUDA)
- Optimized communication primitives
- Proven on Blue Gene supercomputers

### For Modelers
- Separation of concerns: model logic vs. network structure
- Composable, reusable components
- Type-safe interfaces
- Multiple execution modes (debug, profile, release)

## Applications

MGS was designed as a **domain-agnostic framework** for heterogeneous network simulation, with particular strength in biological systems.

### Biological Networks & Circuits

MGS is well-suited for modeling **biological networks** across multiple scales and domains:

#### Systems Biology
- **Gene regulatory networks**: Transcription factors, promoters, regulatory cascades
- **Signaling pathways**: Receptor-ligand interactions, phosphorylation cascades, second messengers
- **Metabolic networks**: Enzyme kinetics, substrate channeling, metabolic flux
- **Protein interaction networks**: Binding dynamics, complex formation, allostery

#### Synthetic Biology  
- **Genetic circuits**: Toggle switches, oscillators, logic gates
- **Cell-free systems**: In vitro transcription-translation networks
- **Engineered pathways**: Biosynthetic circuits, sensors, actuators

#### Neuroscience
- **Neural tissue models**: Multi-scale brain simulations (NTS)
- **Network dynamics**: Oscillations, synchronization, information flow
- **Synaptic plasticity**: Learning rules, homeostasis, development

#### Ecology & Evolution
- **Population dynamics**: Predator-prey, competition, mutualism
- **Evolutionary dynamics**: Fitness landscapes, selection, drift
- **Epidemiology**: Disease spread on contact networks

### Why MGS for Biological Networks?

Biological systems share key characteristics that MGS addresses:

1. **Heterogeneity**: Different component types (genes, proteins, cells, neurons) with distinct dynamics
2. **Multi-scale**: Processes spanning milliseconds to hours, nanometers to centimeters
3. **Complex topology**: Non-uniform connectivity patterns, spatial structure
4. **Large scale**: Thousands to millions of interacting components

MGS's interface-based composition allows you to:
- Mix deterministic and stochastic models
- Couple continuous and discrete dynamics  
- Integrate spatial and temporal scales
- Compose models from different biological levels

**Learn more**: See Justin Bois's [Biological Circuits](https://biocircuits.github.io/) for background on biological network modeling.

### Machine Learning & Deep Neural Networks

MGS can express modern deep learning architectures as explicit computation graphs. The [MNIST example](examples/ml/MNIST.gsl) implements a multi-layer perceptron with:
- **Backpropagation** through the graph
- **Adam optimization** (momentum + RMSprop)
- **Automatic differentiation** via explicit gradient edges
- **Mini-batch training** with supervisor nodes

This demonstrates that contemporary ML architectures map naturally onto MGS's hypergraph abstraction.
```gsl
// Example: 3-layer network structure in GSL
Composite DNN {
  Grid L1 { Dimension(28,28); Layer(nodes, DNNode, UniformLayout(1)); }
  Grid L2 { Dimension(16,16); Layer(nodes, DNNode, UniformLayout(1)); }  
  Grid L3 { Dimension(10,1);  Layer(nodes, DNNode, UniformLayout(1)); }
  
  // Forward connections + backward gradient flow
  l1.nodes -> l2.nodes via DNEdgeSet;
  l2.nodes -> l3.nodes via DNEdgeSet;
  SupervisorNode -> l3.nodes.backward;
}
```

**Technical Note**: [Implementing Backpropagation as Hypergraphs](docs/papers/technical-notes/MNIST_backprop_hypergraph.pdf) describes the architecture in detail.

### Early Self-Supervised Learning

Our 2007 work implemented **topographic infomax** for unsupervised feature learning - predating the modern self-supervised learning paradigm:

> Kozloski J., Cecchi G., Peck C., Rao A. (2007)  
> "Topographic Infomax in a Neural Multigrid"  
> *Advances in Neural Networks* [[PDF](docs/papers/framework/ISNN2007.pdf)]

---

### Example Use Cases by Domain

| Domain | Example System | MGS Components |
|--------|---------------|----------------|
| **Systems Biology** | Repressilator genetic circuit | Gene models, promoter models, mRNA/protein dynamics |
| **Neuroscience** | Cortical column simulation | Neuron models, synapse models, gap junction models |
| **Synthetic Biology** | CRISPR interference circuit | dCas9 models, guide RNA models, gene expression |
| **Machine Learning** | Convolutional neural network | Neuron models, weight models, pooling operations |
| **Epidemiology** | COVID spread on contact network | Individual models, contact models, infection dynamics |

All specified in MDL (models) + GSL (topology) → compiled to optimized parallel C++ code.

## Quick Start

### Prerequisites
```bash
# Ubuntu/Debian
sudo apt-get install build-essential bison flex libgmp-dev python3

# macOS
brew install bison flex gmp python3
```

### Build
```bash
git clone --recursive https://github.com/IBM/MGS.git
cd MGS

# Build for Linux (default target)
./build_mgs -p LINUX --as-MGS

# Or build with GPU support
./build_mgs -p LINUX --as-GPU
```

### Run Example
```bash
# MNIST example (coming soon)
bin/gslparser examples/ml/MNIST.gsl
```

## Documentation

- **[Getting Started Guide](docs/getting-started.md)** - Build, run your first simulation
- **[Technical Documentation](docs/technical-guide.md)** - MDL/GSL language reference
- **[Research Papers](docs/papers/BIBLIOGRAPHY.md)** - Publications describing the framework
- **[API Reference](docs/api/)** - Generated code documentation

## Research & Publications

MGS has been described in multiple peer-reviewed publications. See our [annotated bibliography](docs/papers/BIBLIOGRAPHY.md) for:

- **Framework papers**: Core architecture and design
- **Performance studies**: Scaling on Blue Gene and modern HPC systems  
- **Application papers**: Domain-specific uses

### Key Papers

> Kozloski J., Eleftheriou M., Fitch B., Peck C. (2009)  
> "Interoperable Model Graph Simulator for High-Performance Computing"  
> *IBM Research Report RC24811*  
> [[PDF](docs/papers/framework/IBM_RC_24811.pdf)]

> Kozloski J., Wagner J. (2011)  
> "An Ultrascalable Solution to Large-Scale Neural Tissue Simulation"  
> *Frontiers in Neuroinformatics*  
> [[PDF](docs/papers/framework/fninf0500015.pdf)]

## Project Status

🟢 **Active Development** - MGS is being modernized for contemporary HPC systems:

- ✅ macOS M-series support
- ✅ Modern CUDA integration  
- 🔄 GPU memory management improvements (in progress)
- 🔄 Python bindings (planned)
- 🔄 Cloud deployment examples (planned)

## Contributing

We welcome contributions! Whether you're interested in:
- 🐛 Bug fixes and platform support
- 📚 Documentation improvements  
- 🧪 New example models
- 🚀 Performance optimizations

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/IBM/MGS/issues)
- **Discussions**: [GitHub Discussions](https://github.com/IBM/MGS/discussions)

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

## Acknowledgments

MGS was developed at IBM Research with contributions from James Kozloski, Maria Eleftheriou, Blake Fitch, Charles Peck, and others.

---

**Ready to build large-scale network simulations?** Start with our [Getting Started Guide](docs/getting-started.md) or explore [example models](examples/).
