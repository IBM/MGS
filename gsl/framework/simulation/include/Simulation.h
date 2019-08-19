// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-14-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

// Note: acc. to pthread documentation,
// the pthread.h must be the first included file

#ifndef SIMULATION_H
#define SIMULATION_H
#include "Copyright.h"

#ifndef LINUX
#include <sys/processor.h>
#include <sys/thread.h>
#endif

#ifndef DISABLE_PTHREADS
#include <pthread.h>
//#include <aix41_pthread.h>
#ifdef WITH_XLC
#include <lib_lock.h>
#endif
#endif  //  DISABLE_PTHREADS

#include "SysTimer.h"
#include "Publishable.h"

#include <list>
#include <deque>
#include <memory>
#include <vector>
//#include <cstring>
#include <cstdlib>

#include "TypeManager.h"
#include "ConstantType.h"
#include "VariableType.h"
#include "NodeType.h"
#include "EdgeType.h"
#include "TriggerType.h"
#include "GranuleMapperType.h"
#include "FunctorType.h"
#include "StructType.h"
#include "TypeRegistry.h"
#include "DependencyParser.h"
#include "PhaseElement.h"
#include "GranuleSet.h"
#include "SeparationConstraint.h"
#include "rndm.h"

#ifdef HAVE_MPI
#include "OutputStream.h"
#include "IIterator.h"
#include <mpi.h>
#endif  // HAVE_MPI

#if defined(USE_THREADPOOL_C11)
#include "ThreadPoolC11.h"
#endif

#if defined(REUSE_NODEACCESSORS)
#include "NodeInstanceAccessor.h"
#endif

#include <limits.h>
#include <unordered_map>

class CompCategory;
class Repertoire;
class TriggeredPauseAction;
class CommunicationEngine;
class DistributableCompCategoryBase;
class EdgeCompCategoryBase;

#ifndef DISABLE_PTHREADS
class ThreadPool;
#endif  // DISABLE_PTHREADS

class Pauser;
class Stopper;
class UserInterface;
class WorkUnit;
class Trigger;
class Publisher;
class SimulationPublisher;
class PublisherRegistry;
class InstanceFactoryRegistry;
class NodeSet;
class NDPairList;
class NodeDescriptor;
class GranuleMapper;
class Granule;
class Graph;
class Partitioner;
#ifdef HAVE_MPI
class ISender;
class IReceiver;
#endif

class Simulation : public Publishable {
  friend class SimulationPublisher;
  friend class ISender;
  friend class IReceiver;

  public:
  enum StateType {
    _UNUSED,
    _RUN,
    _STOP,
    _PAUSE,
    _TERMINATE
  };
  enum PassType {
    _GRANULE_MAPPER_PASS,
    _COST_AGGREGATION_PASS,
    _SIMULATE_PASS
  };

#ifndef DISABLE_PTHREADS
  //Simulation(int N, bool bindThreadsToCpus, int numWorkUnits, unsigned seed);
  Simulation(int N, bool bindThreadsToCpus, int numWorkUnits, unsigned seed, int gpuID);
#else // DISABLE_PTHREADS
  //Simulation(int numWorkUnits, unsigned seed);
  Simulation(int numWorkUnits, unsigned seed, int gpuID);
#endif  // DISABLE_PTHREADS

#ifdef HAVE_MPI
  OutputStream* getOutputStream(int pid);
  bool P2P() { return _P2P; }
  bool AllToAllW() { return _alltoallw; }
  bool AllToAllV() { return _alltoallv; }
#endif

  // Functions due to being publishable
  virtual const char* getServiceName(void* data) const;
  virtual const char* getServiceDescription(void* data) const;
  Publisher* getPublisher() { return _publisher; }

  // Compcategory entry point
  void registerCompCat(CompCategory* c);
  void registerDistCompCat(DistributableCompCategoryBase* c);
  void registerEdgeCompCat(EdgeCompCategoryBase* c);

  // Type loaders or retrievers
  TriggerType* getTriggerType(const std::string& typeName) {
    return _triggerRegistry->getType(*this, *_dependencyParser, typeName);
  }
  PublisherRegistry* getPublisherRegistry() { return _publisherRegistry; }
  GranuleMapperType* getGranuleMapperType(const std::string& typeName) {
    return _granuleMapperRegistry->getType(*this, *_dependencyParser, typeName);
  }
  int getNumberOfGranuleMappers() { return _granuleMappers.size(); }
  GranuleMapper* getGranuleMapper(unsigned gmIndex) {
    return _granuleMappers[gmIndex];
  }
  FunctorType* getFunctorType(const std::string& typeName) {
    return _functorRegistry->getType(*this, *_dependencyParser, typeName);
  }
  StructType* getStructType(const std::string& typeName) {
    return _structRegistry->getType(*this, *_dependencyParser, typeName);
  }
  ConstantType* getConstantType(const std::string& typeName) {
    return _constantRegistry->getType(*this, *_dependencyParser, typeName);
  }
  VariableType* getVariableType(const std::string& typeName) {
    return _variableRegistry->getType(*this, *_dependencyParser, typeName);
  }
  NodeType* getNodeType(const std::string& typeName,
                        const NDPairList& ndpList) {
    return _ntm->getType(*this, *_dependencyParser, typeName, ndpList);
  }
  EdgeType* getEdgeType(const std::string& typeName,
                        const NDPairList& ndpList) {
    return _etm->getType(*this, *_dependencyParser, typeName, ndpList);
  }

  DependencyParser* getDependencyParser() { return _dependencyParser; }
  Pauser* getPauser() { return _pauser; }
  Stopper* getStopper() { return _stopper; }
  Repertoire* getRootRepertoire() { return _root; }
  UserInterface* getUI() { return _ui; }
  TriggeredPauseAction* getTriggeredPauseAction() {
    return _triggeredPauseAction;
  }
  const std::vector<InstanceFactoryRegistry*>& getInstanceFactoryRegistries() {
    return _instanceFactoryRegistries;
  }
  //TUAN TODO: change to using size_t
  unsigned getIteration() { return _iteration; }

  std::vector<std::string> const& getPhaseNames() { return _phaseNames; }

  int getNumWorkUnits() { return _numWorkUnits; }

  int getNumGranules() { return _numGranules; }

  float getTime();
  RNG& getWorkUnitRandomSeedGenerator() { return _rng; }
  RNG& getSharedWorkUnitRandomSeedGenerator() { return _rngShared; }
  RNG_ns& getFunctorRandomSeedGenerator() { return _rng; }
  RNG_ns& getSharedFunctorRandomSeedGenerator() { return _rngShared; }
  unsigned getRandomSeed() { return _rngSeed; }
  
  
#ifndef DISABLE_PTHREADS
  int getNumCPUs() { return _numCpus; }
  int getNumThreads() { return _numThreads; }
#else
  int getNumCPUs() { return 1; }
  int getNumThreads() { return 1; }
#endif  // DISABLE_PTHREADS

  bool getPauserStatus() { return _pauserStatus; }
  std::string getName() { return "Simulation"; }
  bool isEdgeRelationalDataEnabled() { return _erd; }

  // Two pass related functions [begin|sgc]

  PassType getPassType() const { return _passType; }
  void setCostAggregationPass() { _passType = _COST_AGGREGATION_PASS; }
  void setSimulatePass() { _passType = _SIMULATE_PASS; 
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
    _currentConnectNodeSet = 0;
#endif
  }
  bool isGranuleMapperPass() const {
    return (_passType == _GRANULE_MAPPER_PASS);
  }
  bool isCostAggregationPass() const {
    return (_passType == _COST_AGGREGATION_PASS);
  }
  bool isSimulatePass() const { return (_passType == _SIMULATE_PASS); }
  unsigned getGranuleMapperCount() { return _granuleMapperCount; }
  void incrementGranuleMapperCount() { _granuleMapperCount++; }

  void incrementGranuleMapperCountOnceForVariable() {
    if (!_variableGranuleMapperAlreadyAdded) {
      _granuleMapperCount++;
      _variableGranuleMapperAlreadyAdded = true;
    }
  }
  void setGraph();
  void setPartitioner(Partitioner* partitioner) { _partitioner = partitioner; }
  Partitioner* getPartitioner() { return _partitioner; }

  // This function resets the effects of parsing the network specification
  // language tree. It is intended to be used by the two pass system.
  void resetInternals();

  void addGranuleMapper(std::unique_ptr<GranuleMapper>& granuleMapper);

  void setVariableGranuleMapperIndex(unsigned idx) {
    _variableGranuleMapperIndex = idx;
  }

  GranuleMapper* getVariableGranuleMapper() {
    return _granuleMappers[_variableGranuleMapperIndex];
  }

  bool hasVariableGranuleMapper() {
    return (_variableGranuleMapperIndex != UINT_MAX);
  }

  unsigned incrementCurrentVariableId() { return _variableGlobalId++; }

  Granule* getGranule(const NodeDescriptor& node);
  Granule* getGranule(const VariableDescriptor& vd);
  Granule* getGranule(const unsigned granuleId);
  void getGranules(NodeSet& nodeSet, GranuleSet& granuleSet);

  void addUnseparableGranuleSet(const GranuleSet& granules);

  void setSeparationGranules();

  // Two pass related functions [end|sgc]

  bool isFinished() {
    return (_state == _STOP);
  };

  // Simulation driving functions
  bool start();
  void pause();
  void resume();
  void stop();
  void run();

  void setUI(UserInterface* ui) { _ui = ui; }
  void disableEdgeRelationalData() { _erd = false; }
  void setPauserStatus(bool pauserStatus) { _pauserStatus = pauserStatus; }
  void addSocket(int fd);
  void addInitPhase(const std::string& name, machineType mType);
  void addRuntimePhase(const std::string& name, machineType mType);
  void addLoadPhase(const std::string& name, machineType mType);
  void addFinalPhase(const std::string& name, machineType mType);

  void addWorkUnits(const std::string& name, std::deque<WorkUnit*>& workUnits);
  void addTrigger(const std::string& name, Trigger* trigger);

  std::string findLaterPhase(const std::string& first,
                             const std::string& second);

  machineType getPhaseMachineType(std::string const & name);
  std::string getFinalRuntimePhaseName();
  void detachUserInterface() { _detachUserInterface = true; }
  virtual ~Simulation();

  int getRank() const { return _rank; }

  int getNumProcesses() const { return _nump; }

  const std::string& getPhaseName() const { return _phaseName; }

  const std::list<CompCategory*>& getCatList() const { return _catList; }

  const std::list<DistributableCompCategoryBase*>& getDistCatList() const {
    return _distCatList;
  }

  const std::list<EdgeCompCategoryBase*>& getEdgeCatList() const {
    return _edgeCatList;
  }

#if defined(HAVE_GPU) 
      // track Granule* with the index of NodeInstanceAccessor
      // as the search from node index to Granule is expensive
      //to save computation time
#if defined(TEST_IDEA_TRACK_GRANULE)
      //we don't need the value of type 'bool', use unordered_map for fast searching
      // std::string = nodetype name such as 'LifeNode'
      //std::unordered_map<Granule*, std::unordered_map<int, bool> > _granule2Nodes;
      //   -> the Granule* and the index of 'LifeNode'-NodeAccessor's associated with that granule
  std::map< std::string, 
      std::unordered_map<Granule*, std::vector<size_t> >> _granule2Nodes;
      //track only Granule* belong to current rank
      //std::vector<Granule*> _granule2Nodes;
#endif

  /* 
   *       keep tracks # nodes created for each nodetype, e.g. LifeNode
   *        on all ranks
   * < "LifeNode", <rank_0 = 1, rank_1 = 2, rank_2 = 5> > 
   */
  std::map< std::string, std::vector<int> > _nodes_count;  //maybe slower
  //std::map< std::string, std::list<int> > _nodes_count; 
  // "LifeNode" on what (original and unique) partionId
  //std::map< std::string, std::vector<int> >  _nodes_partition;
#if defined(DETECT_NODE_COUNT) && (DETECT_NODE_COUNT == NEW_APPROACH)
  //For a given model, e.g. "LifeNode", its node-instances can belong to different Granule
  //  we track how many of node-instances for a given modeltype are assigned to one Granule*
  //  and finally, one we know the Granule* that belongs to the same MPI rank 
  //  --> we can find the total node-instance to be created
  //  THIS METHOD IS FASTER, LESS MEMORY
  std::map< std::string, std::map<Granule*, int> >  _nodes_granules; 
#else
  std::map< std::string, std::vector<Granule*> >  _nodes_granules; //maybe slower
#endif
  //std::map< std::string, std::list<Granule*> >  _nodes_granules; 


  /*
   *     keep tracks # proxy (of the 'from' NodeDescriptor)
   *     to be created for a given proxytype, e.g. CG_LifeNodeProxy
   *     which is supposed to be from different MPI ranks
   *     (i.e. the partition of the granule associated with that 'from' NodeDescriptor is differet from current rank)
   */
  /* 
   * NOTE: Maybe we just need to store 'LifeNode', rather than'CG_LifeNodeProxy
   * < "CG_LifeNodeProxy", 
   *      {0 : <rank_0 = 0, rank_1 = 2, rank_2 = 5> 
   *       1 : <rank_0 = 3, rank_1 = 0, rank_2 = 0> 
   *       2 : <rank_0 = 2, rank_1 = 0, rank_2 = 0> }
   * > 
   */
  //std::map< std::string, std::map< int, std::vector<int> > > _proxy_count;
  /* however, if we're only interested in the current rank
   * i.e. key = current_rank
   */
  std::map< std::string,  std::vector<size_t>  > _proxy_count;  //maybe slower [use size_t]
  //std::map< std::string,  std::vector<int>  > _proxy_count;  //maybe slower
  //std::map< std::string,  std::list<int>  > _proxy_count;
  // representing the connection from nodetype1 to nodetype2
  // NOTE: consider using the 'LifeNode' or 'CG_LifeNodeProxy' for the string
//#if defined(DETECT_PROXY_COUNT) && (DETECT_PROXY_COUNT == NEW_APPROACH)
//  std::map< std::pair<std::string, std::string>, 
//    /* with first Granule* from instances of nodes from nodetype1 are tracked */
//    /* 'int' is the count of node belonging to that Granule */
//    /* with second Granule* from instances of nodes from nodetype2 are tracked */
//    /* 'int' is the count of node belonging to that Granule */
//    std::pair< std::map<Granule*, int>, 
//               std::map<Granule*, int> 
//             > 
//      >  _nodes_from_to_granules;
//#else
//  std::map< std::pair<std::string, std::string>, 
//    /* with Granule* from instances of nodes from nodetype1 are tracked */
//    /* with Granule* from instances of nodes from nodetype2 are tracked */
//    //std::vector< std::pair<std::vector<Granule*>, std::vector<Granule*> > >
//    
//    //std::vector< std::pair<Granule*, Granule* > > //maybe slower
//    std::list< std::pair<Granule*, Granule* > >
//      >  _nodes_from_to_granules;
//#endif
  /* 
  std::map< std::pair<std::string, std::string>, 
    std::vector< std::pair<NodeDescriptor*, NodeDescriptor*> >
      >  _nodes_from_to_ND;
  As 'NodeDescriptor*' are deleted once sim.resetInternal() is called
  we propose using an integer to uniquely pointing
   * */
  //std::map< std::pair<std::string, std::string>, 
  //  std::vector< std::pair<int, int > >
  //    >  _nodes_from_to_ND;
  ////std::vector<NodeDescriptor*> _nodes_ND; //temporary keep tracks of ND has been used
  
  /* Is this correct? Granule* is unique for each layer, but NOT unique for grid-elements 
   *  in that layer
   */
  //std::map<Granule*, std::vector<int>> _granules_and_NDIdentifier;
  // it tells this Granule* (representing what nodetype as the referenced by 'from')
  //    hold the 'LifeNode' how many instance
  //    i.e. it tells how many instance from the rank given by 'Granule*'->getPartitionId()
  std::map<Granule*, std::map<std::string, int>> _granulesFrom_NT_count;
  //std::map<Granule*, std::vector<NodeDescriptor*> > _granulesFrom_and_ND;

#if defined(DETECT_PROXY_COUNT) && (DETECT_PROXY_COUNT == NEW_APPROACH)
  std::unordered_set<NodeDescriptor*> _nodes_ND; //telling if a ND is already used or not
#else
  std::unordered_map<NodeDescriptor*, int> _nodes_ND; //telling if a ND is already used or not
  //std::map<NodeDescriptor*, int> _nodes_ND; //map a ND to an integer, unique for the nodetype
     // that the ND represents
  //std::map<std::string, int> _current_ND_index; //track the current NodeDescriptor for the 
     // nodetype as name given by the key
#endif


    //std::vector<int> > _nodes_count; 
  void print_GPU_info(int devID);

#if defined(SUPPORT_MULTITHREAD_CONNECTION)
#if defined(USE_THREADPOOL_C11)
  std::unique_ptr<ThreadPoolC11> threadPoolC11;
#endif
#endif


#if defined(REUSE_NODEACCESSORS)
  //track nodeaccessors from every model, e.g. CG_LifeNodeGridLayerData
  //IMPORTANT: At each Layer statement, it creates a new CG_LifeNodeGridLayerData object
  //// so each layer has its own array of _nodeInstanceAccessors
  //["LifeNode"][layerIndex] = _nodeInstanceAccessors;
  //NOTE: NodeDescriptor *  is a NodeInstanceAccessor * 
  std::map< std::string, std::map <int, NodeInstanceAccessor*> >
      nodeInstanceAccessor;
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
  //at a given ConnectNodeset, of index 'i'
  // the node of ND 'key' connect to the nodes in std::list
  //keep track the connection from 1 ND (the key) to many other ND
  //std::map< NodeInstanceAccessor*, 
  //  std::map< int, std::list<NodeInstanceAccessor*>>
  //    >
  //    ND_from_to;
  std::unordered_map< int, 
    std::unordered_map< NodeDescriptor*,  /* the node that either connect to or get connection from nodes in the vector below*/ 
      std::pair<std::vector<NodeDescriptor*>, int> > /*  the .second is the integer tracking current index in the vector*/
      >
      ND_from_to;
  int _currentConnectNodeSet; //track the current connection NodeSet? do we need this or 
#endif
  // the ConnectNodeSetFunctor also have the index?
  //std::map< NodeInstanceAccessor*, int>
  //    ND_from_to_currentIndex;
#if defined(REUSE_NODEACCESSORS) and defined(TRACK_SUBARRAY_SIZE)
  //SUBARRAY_DETECTION
  //DETECT subarray-size for each node for a given model
  //A node-instance is uniquely identified by its
  //   1. gridLayerindex
  //   2. the global index in the array _node or the associated _nodeAccessor
  //For a given model, e.g. "LifeNode", its node-instances can belong to different Granule
  //  we track (gridIndex, n) of the nodes on each Granule*
  //  and finally, those Granule* belong to the same MPI rank --> total node-instance to be created
  //std::map< std::string, std::map<Granule*,  std::pair<int, int> > >  _nodes_granules_information; 
  //   #if defined(BASED_ON_INDEX)
  ////--> use "LifeNode": {gridlayerIndex, global-node-index}
  ////NOTE: In NTS, each rank does not know #nodes in the other ranks, so using this is may not working
  //std::map< std::string, std::pair<int, int> >  _nodes_identifier; 
  //   #else
  ////--> use "LifeNode": {gridlayerIndex, associated-NodeAccessor}
  //std::map< std::string, std::pair<int, NodeDescriptor*> >  _nodes_identifier; 
  //   #endif
  // ModelName, then DataMemberOfTypeArray(should be 1D) then size for each node instance
  //  {"LeakyIAFUnit": { "um_inputs": {std::pair<int, int> : int}}
  //  {"LeakyIAFUnit": { "um_inputs": {std::pair<gridlayerIndex, nodeAccessorIndex> : number-of-inputs-for-this-data-member-as-array}}
  //  {"LeakyIAFUnit": { "um_inputs": {std::pair<gridlayerIndex, nodeAccessorIndex> : the-expected-size-for-um_inputs}}
  std::map< std::string, std::map< std::string, std::map<std::pair<int, int> , int> > > _nodes_subarray;  
  //  {"LeakyIAFUnit": one-instance-of-this-class}
  std::map< std::string, Node* >  _nodeShared; 

  //std::map< std::string, std::map< std::string, std::vector<int> > > _nodes_subarray;  
  //std::map< std::string, Node* >  _proxyShared; 
  /* store the current index in flat-flat array */
  /* maybe we don't need this as the um_inputs.size() should be the same value */
  //std::map< std::string, std::map< std::string, size_t> > _nodes_subarray_current_index_count;  
#endif
#endif

#endif
  //DEBUG PURPOSE
  size_t _counter = 0; // for debug purpose, e.g. test how many times a function has been called
  void resetCounter(){_counter=0; };
  void increaseCounter(){++_counter;};
  size_t getCounter(){return _counter;};
  //these two new functions to support printing time info
  void benchmark_start(const std::string&);
  double benchmark_timelapsed(const std::string&);
  void benchmark_timelapsed_diff(const std::string&);
  void benchmark_set_timelapsed_diff();
  void benchmark_end(const std::string&);
  //END DEBUG PURPOSE
  private:
  StateType _state;
  //TUAN TODO: change to using size_t
  unsigned _iteration;
  double currentTime; 
  TypeManager<NodeType>* _ntm;
  TypeManager<EdgeType>* _etm;
  SysTimer _simTimer;
  double _prevTimeElapsed;
  float _mark;
  Repertoire* _root;
  
#ifndef DISABLE_PTHREADS
  // mutex to protect changes to 'state' variable
  pthread_mutex_t _stateMutex;
  pthread_mutex_t _timerMutex;
  pthread_mutex_t _socketsMutex;
#endif  // DISABLE_PTHREADS

  TypeRegistry<TriggerType>* _triggerRegistry;
  PublisherRegistry* _publisherRegistry;
  TypeRegistry<GranuleMapperType>* _granuleMapperRegistry;
  TypeRegistry<FunctorType>* _functorRegistry;
  TypeRegistry<StructType>* _structRegistry;
  TypeRegistry<ConstantType>* _constantRegistry;
  TypeRegistry<VariableType>* _variableRegistry;
  UserInterface* _ui;
  TriggeredPauseAction* _triggeredPauseAction;

#ifndef DISABLE_PTHREADS
  int _numThreads;
  ThreadPool* _threadPool;
  int _numCpus;
#endif  // DISABLE_PTHREADS

  bool _pauserStatus;
  Pauser* _pauser;
  Stopper* _stopper;
  bool _erd;

  // Two pass related members [begin|sgc]

  PassType _passType;
  // It is important to have this separate and not use the arrays size
  // because, this is going to be resetted for the second pass
  unsigned _granuleMapperCount;

  unsigned _variableGranuleMapperIndex;
  unsigned _variableGlobalId;
  bool _variableGranuleMapperAlreadyAdded;

  // This is used to assign a new id to each granule. This is done
  // because the smaller id is used in the algorithm that determines
  // the graph id after dependency resolution.
  unsigned _globalGranuleId;

  // Graph size shows how many vertices will be in the graph, this might
  // be different from the number of granules due to the dependency
  // constraints on nodesets.
  unsigned _graphSize;

  // The graph is used to figure out the memory space to which granules belong.
  Graph* _graph;

  std::vector<GranuleMapper*> _granuleMappers;

  std::list<SeparationConstraint*> _separationConstraints;

  // Can be used like this, only because it will be resized at the
  // beginning, we couldn't do this if something was added later on.
  std::vector<Granule> _separationGranules;

  // Two pass related members [end|sgc]

  // Publisher is constructed in Simulation constructor and ownership
  // handed off to PublisherRegistry
  Publisher* _publisher;
  // The Dynamic (shared object) loader
  DependencyParser* _dependencyParser;
  bool _detachUserInterface;

  std::vector<InstanceFactoryRegistry*> _instanceFactoryRegistries;
  std::list<CompCategory*> _catList;
  std::list<DistributableCompCategoryBase*> _distCatList;
  std::list<EdgeCompCategoryBase*> _edgeCatList;
  std::list<int> _socketsInUse;
  std::deque<PhaseElement> _initPhases;
  std::deque<PhaseElement> _runtimePhases;
  std::deque<PhaseElement> _loadPhases;
  std::deque<PhaseElement> _finalPhases;
  std::map<std::string, bool> _communicatingPhases;
  //map from a simulation phase's name, e.g. 'update' (as declared in GSL), to the machine type, e.g. GPU, which handle that type
  std::map<std::string, machineType> _machineTypes;
  int _rank;
  int _nump;
  std::string _phaseName;

#ifdef HAVE_MPI
  std::vector<OutputStream*> _outputStreams;
  std::map<int, int> _pidsVsOrders;
  static int P2P_TAG;

  IIterator<ISender>* _iSenders;
  IIterator<IReceiver>* _iReceivers;
  CommunicationEngine* _commEngine;

  bool _P2P;
  bool _alltoallw;
  bool _alltoallv;

#endif  // HAVE_MPI
  std::vector<std::string> _phaseNames;

  void pauseHandler();
  void resumeHandler();
  void stopHandler();
  void updateAll();
  inline void runPhases(std::deque<PhaseElement>& phases);
  PhaseElement& getPhaseElement(const std::string& name);
  RNG _rng;
  RNG _rngShared;
  unsigned _rngSeed;  
  int _numWorkUnits;
  int _numGranules;
  Partitioner* _partitioner;
//#define TEST_PUTTING_nodes_toSimulation
//#if defined(HAVE_GPU)  and defined(TEST_PUTTING_nodes_toSimulation)
//  //for testing purpose
//  //to see if allocate here still slow
//  public:
//  //std::map<std::string,
//  //    ShallowArray_Flat<LifeNode, Array_Flat<int>::MemLocation::CPU, 1000>
//  //      > _nodes;
//  ShallowArray_Flat<LifeNode, Array_Flat<int>::MemLocation::CPU, 1000> _nodes;
//#endif

};

#endif
