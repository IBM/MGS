# Comprehensive Interoperable Model Graph Simulator Framework Guide

## Table of Contents

### Part I: Framework Architecture and Languages
1. [Architecture Overview](#1-architecture-overview)
   - [1.1 Key Concepts](#11-key-concepts)
   - [1.2 Execution Rules](#12-execution-rules)
2. [Model Definition Language (MDL)](#2-model-definition-language-mdl)
   - [2.1 Core Elements](#21-core-elements)
   - [2.2 Key Syntax Features](#22-key-syntax-features)
   - [2.3 Connection Mechanism](#23-connection-mechanism)
3. [Graph Specification Language (GSL)](#3-graph-specification-language-gsl)
   - [3.1 Core Elements](#31-core-elements)
   - [3.2 Key Syntax Features](#32-key-syntax-features)
   - [3.3 Connection Patterns](#33-connection-patterns)
4. [Neural Network Implementation Example](#4-neural-network-implementation-example)
   - [4.1 Core Components](#41-core-components)
   - [4.2 Network Architecture](#42-network-architecture)
   - [4.3 Learning Process](#43-learning-process)

### Part II: Implementation and Code Generation
5. [Code Generation Overview](#5-code-generation-overview)
   - [5.1 Generation Process](#51-generation-process)
   - [5.2 Directory Structure](#52-directory-structure)
   - [5.3 File Naming Conventions](#53-file-naming-conventions)
6. [Generated Code Structure](#6-generated-code-structure)
   - [6.1 Base Classes and Prefixes](#61-base-classes-and-prefixes)
   - [6.2 Component Categories](#62-component-categories)
   - [6.3 Parameter Sets](#63-parameter-sets)
   - [6.4 Work Units](#64-work-units)
7. [Interface Implementation](#7-interface-implementation)
   - [7.1 Interface Methods](#71-interface-methods)
   - [7.2 Interface Matching](#72-interface-matching)
8. [Connection Implementation](#8-connection-implementation)
   - [8.1 Connection Methods](#81-connection-methods)
   - [8.2 Predicates and Pattern Matching](#82-predicates-and-pattern-matching)
   - [8.3 Data Mapping](#83-data-mapping)
9. [Execution Model](#9-execution-model)
   - [9.1 Initialization](#91-initialization)
   - [9.2 Phase Execution](#92-phase-execution)
   - [9.3 Communication](#93-communication)
10. [Parallel Computing Support](#10-parallel-computing-support)
    - [10.1 MPI Implementation](#101-mpi-implementation)
    - [10.2 GPU Acceleration](#102-gpu-acceleration)
    - [10.3 Memory Management Strategies](#103-memory-management-strategies)
11. [Creating New Models](#11-creating-new-models)
    - [11.1 Development Workflow](#111-development-workflow)
    - [11.2 Customizing Templates](#112-customizing-templates)
    - [11.3 Best Practices](#113-best-practices)

---

## Part I: Framework Architecture and Languages

### 1. Architecture Overview

The Interoperable Model Graph Simulator is a system designed by IBM Research (2009) for composing and executing large-scale heterogeneous network simulations, with a focus on computational neuroscience applications. The framework provides:

- A programming model for defining interoperable models that can communicate with each other
- A means for declaring graphs where vertices are instances of these models
- An execution architecture that supports parallel computing on distributed systems like IBM's Blue Gene

#### 1.1 Key Concepts

- **Models**: Basic computational units with encapsulated state, interfaces for communication, and execution phases
- **Graphs**: Networks composed of model instances (vertices) and their connections (edges)
- **Execution Phases**: Sequential processes that organize computation and communication
- **Granules**: Groupings of models for efficient partitioning across computing resources

#### 1.2 Execution Rules

1. **Acyclic Rule**: Models sharing an execution phase must not have data dependencies
2. **Data Locality Rule**: Models only access local data through defined interfaces
3. **Proxy Communication Rule**: Communication between models occurs only at phase boundaries

### 2. Model Definition Language (MDL)

MDL is a domain-specific language for defining the types of models that can be instantiated in a simulation.

#### 2.1 Core Elements

- **Structs**: Define composite data types
- **Interfaces**: Define communication contracts between models
- **Models**: Define computational elements that implement interfaces
- **Phases**: Define execution stages for models
- **Connections**: Define how models interact with other models

#### 2.2 Key Syntax Features

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

#### 2.3 Connection Mechanism

- Models define what interfaces they implement (provide)
- Models define what interfaces they expect from other models
- Connections are established through predicates that test compatibility
- Interface data is mapped to internal model state

### 3. Graph Specification Language (GSL)

GSL is a companion language to MDL that focuses on instantiating models and organizing them into networks.

#### 3.1 Core Elements

- **Grids**: Spatial organizations of model instances
- **Layers**: Groupings of models within grids
- **Connection Functors**: Patterns for establishing connections
- **Composites**: Hierarchical combinations of grids and other components
- **Phases**: Execution phase declarations and mappings

#### 3.2 Key Syntax Features

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

#### 3.3 Connection Patterns

GSL provides rich mechanisms for defining connection patterns between models:
- Direct one-to-one connections
- Spatial sampling patterns (e.g., radial sampling)
- Many-to-many connections with various distribution functions

### 4. Neural Network Implementation Example

The framework supports neural network implementations, as demonstrated in the MNIST example.

#### 4.1 Core Components

- **Neurons**: Implemented as DNNode instances
- **Weights**: Implemented as DNEdgeSet instances
- **Layers**: Organized as Grids with dimensions matching network requirements
- **Supervisor**: Handles training data and learning signal propagation

#### 4.2 Network Architecture

- Multi-layer neural network for MNIST digit classification
- Input layer: 28×28 (matches MNIST images)
- Hidden layers with various connection patterns
- Output layer: 10 neurons (one per digit)
- Bias connections for each layer

#### 4.3 Learning Process

- Forward pass: Activation propagation through the network
- Backward pass: Gradient propagation for learning
- Adam optimization (combination of momentum and RMSprop)
- Supervised learning managed by SupervisorNode

---

## Part II: Implementation and Code Generation

### 5. Code Generation Overview

The framework uses a code generation approach to translate high-level MDL and GSL specifications into executable C++ code.

#### 5.1 Generation Process

The code generation process involves:

1. Parsing MDL files using the `mdlparser` command
2. Generating a set of C++ header and implementation files
3. Creating template files for user customization
4. Organizing generated files into a directory structure

The process separates framework infrastructure code (which handles the complexity of execution, memory management, and communication) from model-specific logic (which users implement).

#### 5.2 Directory Structure

When `mdlparser` processes an MDL file like `DNN.mdl`, it generates a directory structure like:

```
DNN-dir/
├── DNEdgeSet/            # Implementation for DNEdgeSet model
├── DNNode/               # Implementation for DNNode model
│   ├── include/          # Header files for DNNode
│   │   ├── CG_DNNode.h   # Generated base class
│   │   ├── DNNode.h.gen  # Template for customization
│   │   └── ...
│   └── src/              # Implementation files
│       ├── CG_DNNode.C   # Generated implementation
│       ├── DNNode.C.gen  # Template for customization
│       └── ...
├── SupervisorNode/       # Implementation for SupervisorNode model
└── ...
```

Each model defined in the MDL file gets its own subdirectory containing the generated code.

#### 5.3 File Naming Conventions

The framework uses specific naming conventions:

- Files with `CG_` prefix: Fully generated code that should not be modified
- Files with `.gen` suffix: Template files intended for developer customization
- Base files without special prefixes/suffixes: Final implementation files

### 6. Generated Code Structure

#### 6.1 Base Classes and Prefixes

For each model defined in MDL, the code generator creates:

1. A base class with `CG_` prefix that implements the framework integration
2. A derived class template (`.gen` files) for custom implementation

For example, for the `DNNode` model:
- `CG_DNNode` class contains all the framework integration code
- `DNNode` class (to be implemented in `.gen` files) inherits from `CG_DNNode` and provides model-specific behavior

```cpp
// In CG_DNNode.h (generated)
class CG_DNNode : public BackwardProducer, public ForwardProducer, public NodeBase {
    // Framework integration
};

// In DNNode.h.gen (template for customization)
class DNNode : public CG_DNNode {
    // Custom implementation
};
```

#### 6.2 Component Categories

For each model, the framework generates a "Component Category" class that manages collections of model instances:

```cpp
// In CG_DNNodeCompCategory.h
class CG_DNNodeCompCategory : public NodeCompCategoryBase {
    // Methods for managing collections of DNNode instances
    // Memory management, execution scheduling, etc.
};
```

Component categories handle:
- Model instance allocation
- Execution scheduling
- Memory management
- Parallel computing integration

#### 6.3 Parameter Sets

The framework generates parameter set classes for handling model parameters and connection attributes:

```cpp
// In CG_DNNodePSet.h
class CG_DNNodePSet : public ParameterSet {
    // Model parameters
    double output;
    double gradient;
    ShallowArray<EdgeSetInput> inputs;
    // ...
};

// In CG_DNNodeInAttrPSet.h
class CG_DNNodeInAttrPSet : public ParameterSet {
    // Connection attributes for incoming connections
    String identifier;
    unsigned index;
    // ...
};
```

These parameter sets:
- Store model state and configuration
- Provide serialization for distributed computing
- Handle attribute matching for connections

#### 6.4 Work Units

The framework generates classes for execution scheduling:

```cpp
// In CG_DNNodeWorkUnitInstance.h
class CG_DNNodeWorkUnitInstance : public WorkUnit {
    // Execution scheduling unit
};
```

Work units:
- Represent units of computation for scheduling
- Provide random number generators for stochastic models
- Support parallel execution across CPU cores and GPU devices

### 7. Interface Implementation

#### 7.1 Interface Methods

The framework generates methods to implement the interfaces defined in MDL:

```cpp
// In CG_DNNode.h and CG_DNNode.C
virtual double* CG_get_BackwardProducer_backward() {
#ifdef HAVE_GPU
    return &(_container->um_gradient[index]);
#else
    return &gradient;
#endif
}

virtual double* CG_get_ForwardProducer_forward() {
#ifdef HAVE_GPU
    return &(_container->um_output[index]);
#else
    return &output;
#endif
}
```

These methods:
- Expose model state variables through interfaces
- Handle different memory management strategies for CPU and GPU
- Implement the interface contracts defined in MDL

#### 7.2 Interface Matching

When establishing connections, the framework performs interface matching:

```cpp
// In CG_DNNode.C
void CG_DNNode::addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) {
    // Check if the node implements ForwardArrayProducer
    ForwardArrayProducer* CG_ForwardArrayProducerPtr = 
        dynamic_cast<ForwardArrayProducer*>(CG_node->getNode());
    
    // Match against connection predicates
    if (CG_castedPSet->identifier == "input") {
        // Connect if interface matches and predicate is satisfied
        // ...
    }
}
```

This implements the interface matching behavior specified in MDL connection declarations.

### 8. Connection Implementation

#### 8.1 Connection Methods

The framework generates methods to handle connections between models:

```cpp
// In CG_DNNode.C
void CG_DNNode::addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) {
    // Handle incoming connections
}

void CG_DNNode::addPostNode(NodeDescriptor* CG_node, ParameterSet* CG_pset) {
    // Handle outgoing connections
}
```

These methods:
- Are called during network construction
- Establish data connections between model instances
- Perform interface matching and predicate evaluation

#### 8.2 Predicates and Pattern Matching

The connection methods implement the predicates defined in MDL:

```cpp
// In CG_DNNode.C
if (CG_castedPSet->identifier == "input") {
    // This implements the predicate (PSet.identifier=="input") from MDL
    // ...
}
```

This allows models to selectively accept connections based on attributes.

#### 8.3 Data Mapping

Once connections are established, the framework sets up data mappings:

```cpp
// In CG_DNNode.C
// This implements the ForwardArrayProducer.forwardArray >> inputs.inputArray mapping
inputs[CG_inputsSize].inputArray = 
    CG_ForwardArrayProducerPtr->CG_get_ForwardArrayProducer_forwardArray();
```

These mappings:
- Link source model outputs to destination model inputs
- Can be direct references (pointer assignments) or proxy mechanisms for distributed computing
- Implement the data flow specified in MDL connection mappings

### 9. Execution Model

#### 9.1 Initialization

The framework implements a two-phase initialization process:

1. **Model Allocation**: Models are instantiated and organized into grid structures
2. **Connection Establishment**: Connections are set up between models
3. **Initialization Phase**: The `initialize()` method is called on each model instance

```cpp
// In CG_DNNodeCompCategory.C
void CG_DNNodeCompCategory::CG_InstancePhase_initialize(
    NodePartitionItem* arg, CG_DNNodeWorkUnitInstance* wu) {
    // Call initialize() on each model instance in a partition
    for (; it <= end; ++it) {
        (*it).initialize(wu->getRNG());
    }
}
```

#### 9.2 Phase Execution

The runtime phases defined in MDL are implemented as work unit executions:

```cpp
// In CG_DNNodeCompCategory.C
void CG_DNNodeCompCategory::CG_InstancePhase_update(
    NodePartitionItem* arg, CG_DNNodeWorkUnitInstance* wu) {
    // Call update() on each model instance in a partition
    for (; it <= end; ++it) {
        (*it).update(wu->getRNG());
    }
}
```

The framework:
- Schedules phases according to the GSL specification
- Ensures data dependencies are respected
- Handles parallel execution across available resources

#### 9.3 Communication

The framework implements communication between models:

1. **Local Communication**: Direct method calls for models in the same process
2. **Distributed Communication**: Proxy objects and MPI for models across processes
3. **Phase Synchronization**: Ensures data is consistent at phase boundaries

```cpp
// In CG_DNNodeProxy.C (for distributed computing)
void CG_DNNodeProxy::CG_recv_update_demarshaller(
    std::unique_ptr<CG_DNNodeProxyDemarshaller> &ap) {
    // Communication mechanism for distributed computing
}
```

### 10. Parallel Computing Support

#### 10.1 MPI Implementation

The framework provides extensive support for MPI-based distributed computing:

```cpp
// In CG_DNNodeCompCategory.C
void CG_DNNodeCompCategory::addToSendMap(int toPartitionId, Node* node) {
    // Add node to send list for communication with specified partition
}

void CG_DNNodeCompCategory::send(int pid, OutputStream* os) {
    // Send node data to specified process
}
```

This allows:
- Running simulations across multiple compute nodes
- Automatic data synchronization between processes
- Load balancing across available resources

#### 10.2 GPU Acceleration

The framework supports GPU acceleration through CUDA:

```cpp
// In DNNodeCompCategory.cu.gen
void __global__ DNNode_kernel_update(
    double* output, double* gradient, /* other parameters */,
    unsigned size) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < size) {
        // GPU implementation of model update
    }
}
```

The GPU support:
- Conditionally compiles based on `HAVE_GPU` macro
- Uses different memory management strategies for GPU memory
- Provides kernel templates for model-specific GPU implementations

#### 10.3 Memory Management Strategies

The framework implements several memory management strategies:

```cpp
// In CG_DNNodeCompCategory.C
#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3
    um_inputs.increaseSizeTo(sz);
#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4
    um_inputs.increaseSizeTo(sz*MAX_SUBARRAY_SIZE);
    um_inputs_start_offset.increaseSizeTo(sz);
    um_inputs_num_elements.increaseSizeTo(sz);
#endif
```

These strategies:
- Optimize memory access patterns for different hardware architectures
- Support different memory layouts for CPU and GPU
- Allow tuning for specific simulation requirements

### 11. Creating New Models

#### 11.1 Development Workflow

To create a new model, follow this workflow:

1. **Define the model in MDL**:
   ```
   Node MyModel Implements SomeInterface {
       // State variables
       double var1;
       
       // Phases
       InitPhase initialize();
       RuntimePhase update(var1);
       
       // Interface mappings
       SomeInterface.method << &var1;
       
       // Connections
       Connection Pre Node (PSet.identifier=="input") Expects OtherInterface {
           // Mappings
       }
   }
   ```

2. **Generate code using mdlparser**:
   ```
   mdlparser MyModel.mdl
   ```

3. **Implement custom behavior** in the generated `.gen` files:
   ```cpp
   // In MyModel.C.gen
   void MyModel::initialize(RNG& rng) {
       // Custom initialization logic
   }
   
   void MyModel::update(RNG& rng) {
       // Custom update logic
   }
   ```

4. **Compile and link** the implementation files with the framework libraries

#### 11.2 Customizing Templates

When customizing the `.gen` template files, focus on:

1. **initialize() method**: Set up initial model state
   ```cpp
   void MyModel::initialize(RNG& rng) {
       // Initialize state variables
       output = 0.0;
       // Use RNG for stochastic initialization if needed
       if (rng.uniform() > 0.5) {
           // ...
       }
   }
   ```

2. **update() method**: Implement model behavior
   ```cpp
   void MyModel::update(RNG& rng) {
       // Access input connections
       double sum = 0.0;
       for (int i = 0; i < inputs.size(); i++) {
           // Process inputs
           sum += *(inputs[i].inputArray);
       }
       
       // Update model state
       output = activation_function(sum);
       
       // Compute gradient for learning
       gradient = compute_gradient();
   }
   ```

3. **extractInputIndex() method**: Custom connection logic
   ```cpp
   void MyModel::extractInputIndex(/* parameters */) {
       // Custom connection logic
       // This is called when connections are established
   }
   ```

#### 11.3 Best Practices

When developing new models:

1. **Understand the MDL-to-code mapping**:
   - State variables become member variables
   - Interfaces become virtual methods
   - Connections become method calls

2. **Handle both CPU and GPU execution**:
   - Be aware of memory management differences
   - Use conditional compilation for platform-specific code
   - Test on both platforms if available

3. **Consider performance implications**:
   - Keep computation local to minimize communication
   - Parallelize appropriately for your model
   - Profile and optimize critical sections

4. **Maintain separation of concerns**:
   - Only modify the `.gen` files
   - Don't alter the generated `CG_` files
   - Focus on model-specific logic, not framework infrastructure

---

This document provides a comprehensive guide to understanding and extending the Interoperable Model Graph Simulator framework, from high-level architecture through implementation details and developer workflows.
