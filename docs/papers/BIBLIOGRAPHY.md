# MGS/NTS Research Bibliography

An annotated guide to publications describing the Model Graph Simulator framework and its applications.

## Framework & Architecture

### Core Framework Papers

**Interoperable Model Graph Simulator for High-Performance Computing** (2009)  
*James Kozloski, Maria Eleftheriou, Blake Fitch, Charles Peck*  
IBM Research Report RC24811  
📄 [[PDF](framework/IBM_RC_24811.pdf)]

> **Abstract**: Describes the core MGS architecture, including the MDL/GSL languages, execution model, and parallel computing strategies. Essential reading for understanding framework design.
>
> **Key Contributions**:
> - Domain-specific languages for model definition and graph specification
> - Interface-based model composition
> - Distributed execution on Blue Gene systems
> - Performance scaling results

---

**An Ultrascalable Solution to Large-Scale Neural Tissue Simulation** (2011)  
*James Kozloski, John Wagner*  
Frontiers in Neuroinformatics 5:15  
📄 [[PDF](framework/fninf0500015.pdf)] | 🔗 [[DOI](https://doi.org/10.3389/fninf.2011.00015)]

> **Abstract**: Presents the Neural Tissue Simulator (NTS) built on MGS, demonstrating framework scalability and flexibility for large-scale simulations.
>
> **Key Contributions**:
> - Scalability analysis on Blue Gene systems
> - Biological model library for neuroscience
> - Performance benchmarks
> - Software engineering practices for scientific computing

---

**Applications of Massively Parallel Systems** (2008)  
*IBM Journal of Research and Development, Volume 52, Issue 1/2*  
📄 [[PDF](framework/IBMJRD_2008.pdf)]

> **Context**: Special issue on Blue Gene applications including MGS/NTS work. Covers the broader ecosystem of massively parallel scientific computing.

---

### Early Work

**Parallel Computing Symposium 2003** (2003)  
*International Conference on Computational Science (ICCS)*  
📄 [[PDF](framework/ICCS_2003.pdf)]

> **Historical Context**: Early computational foundations that influenced MGS design, predating the full framework development.

---

## Machine Learning Applications

### Supervised Deep Learning

**MNIST Classification Network** (2024)  
*James Kozloski*  
Technical Note (Unpublished)  
📄 [[PDF](technical-notes/MNIST_backprop_hypergraph.pdf)] | Code: [[MNIST.gsl](../../examples/ml/MNIST.gsl)]

> **Description**: Complete implementation of a multi-layer perceptron for MNIST digit classification using MGS/GSL. Demonstrates that modern deep learning architectures (backprop, Adam optimization, mini-batch training) can be expressed as explicit hypergraphs in the MGS framework.
>
> **Technical Highlights**:
> - Forward pass: activation propagation through layers
> - Backward pass: gradient computation and backpropagation
> - Adam optimizer: momentum and RMSprop updates
> - Supervisor node: manages training data and learning signals
>
> **Significance**: Shows MGS is not limited to biologically-inspired models but can express contemporary ML architectures. The explicit graph representation makes the computation structure transparent.

---

### Self-Supervised Learning

**Topographic Infomax in a Neural Multigrid** (2007)  
*James Kozloski, Guillermo Cecchi, Charles Peck, A. Ravishankar Rao*  
Lecture Notes in Computer Science, Volume 4492  
📄 [[PDF](framework/ISNN2007.pdf)] | 🔗 [[DOI](https://doi.org/10.1007/978-3-540-72383-7_59)]

> **Abstract**: Implements information maximization (infomax) as an unsupervised learning objective in a multi-scale neural architecture. This work predated the modern renaissance of self-supervised learning.
>
> **Key Contributions**:
> - Information-theoretic learning objective
> - Topographic organization emerges from learning
> - Multi-scale processing (neural multigrid)
>
> **Historical Note**: This 2007 work on self-supervised representation learning anticipated techniques that became central to modern deep learning (2015+).

---

## Biological Network Applications

### Computational Neuroscience

**Striatal Network Modeling in Huntington's Disease** (2020)  
*Adam Ponzi, Scott J. Barton, Kendra D. Brunner, Claudia Rangel-Barajas, Emily S. Zhang, Benjamin R. Miller, George V. Rebec, James Kozloski*  
PLOS Computational Biology 16(4): e1007648  
📄 [[PDF](applications/PLOSCompBio2020Ponzi_et_al.pdf)] | 🔗 [[DOI](https://doi.org/10.1371/journal.pcbi.1007648)]

> **Domain**: Neuroscience application
>
> **Demonstrates**: How MGS enables large-scale biologically detailed network models. Uses medium spiny neuron (MSN) network models with physiologically detailed parameters estimated from single-unit recordings.
>
> **Key Results**:
> - Network dynamics emerge from local MSN interactions
> - Disease phenotypes reproduced across multiple mouse models
> - Framework handles heterogeneous model types and scales

---

**Delta-Gamma Phase-Amplitude Coupling Analysis** (2018)  
*Emily Zhang, et al.*  
eNeuro 5(6): ENEURO.0210-18.2018  
📄 [[PDF](applications/ENEURO_021018_2018_full.pdf)] | 🔗 [[DOI](https://doi.org/10.1523/ENEURO.0210-18.2018)]

> **Domain**: Neurophysiology/electrophysiology
>
> **Demonstrates**: MGS models reproduce experimental observations of oscillatory activity patterns in striatum and cortex.
>
> **Relevance**: Shows framework can capture complex temporal dynamics emerging from network interactions.

---

## Conference Presentations

**Cosyne Poster** (Year TBD)  
📄 [[PDF](presentations/CosyneposterII70.pdf)]

> Computational neuroscience conference presentation.

---

**Database Research Conference** (2007)  
📄 [[PDF](presentations/DBF2007.pdf)]

> Database and data management aspects of large-scale simulations.

---

## Using This Bibliography

### For Framework Understanding
Start with:
1. IBM RC 24811 (2009) - Core architecture
2. Frontiers paper (2011) - NTS application & scalability

### For Applications
- **Machine Learning**: MNIST technical note (2024)
- **Neuroscience**: PLOS Comp Bio (2020), eNeuro (2018)
- **General modeling**: Start with framework papers, then domain-specific

### For Implementation
- Framework papers describe MDL/GSL syntax
- See [technical documentation](../technical-guide.md) for current API

---

## Citation

If you use MGS in your research, please cite:
```bibtex
@techreport{kozloski2009mgs,
  title={Interoperable Model Graph Simulator for High-Performance Computing},
  author={Kozloski, James and Eleftheriou, Maria and Fitch, Blake and Peck, Charles},
  institution={IBM Research},
  number={RC24811},
  year={2009}
}

@article{kozloski2011ultrascalable,
  title={An Ultrascalable Solution to Large-Scale Neural Tissue Simulation},
  author={Kozloski, James and Wagner, John},
  journal={Frontiers in Neuroinformatics},
  volume={5},
  pages={15},
  year={2011},
  doi={10.3389/fninf.2011.00015}
}
```

---

*Last updated: December 2024*
