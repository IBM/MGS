// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
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

#include <limits.h>

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
  Simulation(int N, bool bindThreadsToCpus, int numWorkUnits, unsigned seed);
#else // DISABLE_PTHREADS
  Simulation(int numWorkUnits, unsigned seed);
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
  void setSimulatePass() { _passType = _SIMULATE_PASS; }
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

  void addGranuleMapper(std::auto_ptr<GranuleMapper>& granuleMapper);

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
  void addInitPhase(const std::string& name, PhaseElement::machineType mType);
  void addRuntimePhase(const std::string& name, PhaseElement::machineType mType);
  void addLoadPhase(const std::string& name, PhaseElement::machineType mType);
  void addFinalPhase(const std::string& name, PhaseElement::machineType mType);

  void addWorkUnits(const std::string& name, std::deque<WorkUnit*>& workUnits);
  void addTrigger(const std::string& name, Trigger* trigger);

  std::string findLaterPhase(const std::string& first,
                             const std::string& second);

  PhaseElement::machineType getPhaseMachineType(std::string& name);
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

  private:
  StateType _state;
  unsigned _iteration;
  TypeManager<NodeType>* _ntm;
  TypeManager<EdgeType>* _etm;
  SysTimer _simTimer;
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
  std::map<std::string, PhaseElement::machineType> _machineTypes;
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
};

#endif
