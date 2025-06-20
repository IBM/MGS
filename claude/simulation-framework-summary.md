# Interoperable Model Graph Simulator Framework
**Summary Document v1.0**

## 1. Architecture Overview

The Interoperable Model Graph Simulator is a system designed by IBM Research (2009) for composing and executing large-scale heterogeneous network simulations, with a focus on computational neuroscience applications. The framework provides:

- A programming model for defining interoperable models that can communicate with each other
- A means for declaring graphs where vertices are instances of these models
- An execution architecture that supports parallel computing on distributed systems like IBM's Blue Gene

### 1.1 Key Concepts

- **Models**: Basic computational units with encapsulated state, interfaces for communication, and execution phases
- **Graphs**: Networks composed of model instances (vertices) and their connections (edges)
- **Execution Phases**: Sequential processes that organize computation and communication
- **Granules**: Groupings of models for efficient partitioning across computing resources

### 1.2 Execution Rules

1. **Acyclic Rule**: Models sharing an execution phase must not have data dependencies
2. **Data Locality Rule**: Models only access local data through defined interfaces
3. **Proxy Communication Rule**: Communication between models occurs only at phase boundaries

## 2. Model Definition Language (MDL)

MDL is a domain-specific language for defining the types of models that can be instantiated in a simulation.

### 2.1 Core Elements

- **Structs**: Define composite data types
- **Interfaces**: Define communication contracts between models
- **Models**: Define computational elements that implement interfaces
- **Phases**: Define execution stages for models
- **Connections**: Define how models interact with other models

### 2.2 Key Syntax Features

```
// Interface definition example
Interface ForwardProducer {
  double* forward;
}

// Model definition example
Node DNNode Implements ForwardProducer, BackwardProducer {
   double output;
   double gradient;
   // More state variables
   
   InitPhase initialize();
   RuntimePhase update(output, gradient);
   
   ForwardProducer.forward << &output;
   BackwardProducer.backward << &gradient;
   
   // Connection definitions
   Connection Pre Node (PSet.identifier=="input") Expects ForwardArrayProducer {
      // Interface mappings
   }
}
```

### 2.3 Connection Mechanism

- Models define what interfaces they implement (provide)
- Models define what interfaces they expect from other models
- Connections are established through predicates that test compatibility
- Interface data is mapped to internal model state

## 3. Graph Specification Language (GSL)

GSL is a companion language to MDL that focuses on instantiating models and organizing them into networks.

### 3.1 Core Elements

- **Grids**: Spatial organizations of model instances
- **Layers**: Groupings of models within grids
- **Connection Functors**: Patterns for establishing connections
- **Composites**: Hierarchical combinations of grids and other components
- **Phases**: Execution phase declarations and mappings

### 3.2 Key Syntax Features

```
// Phase declaration example
InitPhases = { initializeShared, initialize };
RuntimePhases = { updateMNIST, updateNodes, updateEdgeSets, lastPhase };

// Node type mapping example
NodeType DNNode() {update->updateNodes};

// Grid definition example
Grid L1 {
   Dimension(28,28);
   Layer(nodes, DNNode, UniformLayout(1));
   // Layer connections
}

// Composite definition example
Composite DNN {
  L1 l1;
  L2 l2;
  // Layer connections
}
```

### 3.3 Connection Patterns

GSL provides rich mechanisms for defining connection patterns between models:
- Direct one-to-one connections
- Spatial sampling patterns (e.g., radial sampling)
- Many-to-many connections with various distribution functions

## 4. Neural Network Implementation Example

The framework supports neural network implementations, as demonstrated in the MNIST example.

### 4.1 Core Components

- **Neurons**: Implemented as DNNode instances
- **Weights**: Implemented as DNEdgeSet instances
- **Layers**: Organized as Grids with dimensions matching network requirements
- **Supervisor**: Handles training data and learning signal propagation

### 4.2 Network Architecture

- Multi-layer neural network for MNIST digit classification
- Input layer: 28×28 (matches MNIST images)
- Hidden layers with various connection patterns
- Output layer: 10 neurons (one per digit)
- Bias connections for each layer

### 4.3 Learning Process

- Forward pass: Activation propagation through the network
- Backward pass: Gradient propagation for learning
- Adam optimization (combination of momentum and RMSprop)
- Supervised learning managed by SupervisorNode

## 5. Implementation Details

The framework employs:

- **Code Generation**: MDL and GSL are compiled to C++ code
- **Runtime System**: Handles execution scheduling and communication
- **Parallelization**: Distributes computation across available resources
- **Communication Optimization**: Efficiently manages data exchange between processes

## 6. References to Detailed Documentation

- **MDL Language Specification**: Comprehensive syntax and semantics
- **GSL Language Specification**: Grid and connection patterns
- **Neural Network Models**: Specific implementation details
- **Execution Architecture**: Runtime behavior and optimization

---

*This summary document will be updated as additional information is incorporated into the project knowledge base.*
