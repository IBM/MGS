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

#ifdef HAVE_MPI
#include <mpi.h>
#include "MpiOutputStream.h"
#endif // HAVE_MPI

#include "Simulation.h"
#ifdef HAVE_MPI
// I selected 21 randomly... --sgc
int Simulation::P2P_TAG = 21;
#endif // HAVE_MPI
#include "TriggeredPauseAction.h"
#include "PauseActionable.h"
#include "Trigger.h"
#include "Pauser.h"
#include "Stopper.h"
//#include "CompCategory.h"
#include "DistributableCompCategoryBase.h"
#include "EdgeCompCategoryBase.h"
#include "TriggerType.h"
#include "FunctorType.h"
#include "StructType.h"
#include "ConstantType.h"
#include "VariableType.h"
#include "TypeRegistry.h"
#include "PublisherRegistry.h"
#include "SysTimer.h"
#include "SimulationPublisher.h"
#include "WorkUnit.h"
#include "Repertoire.h"
#include "Publisher.h"
#include "Service.h"
#include "UserInterface.h"
#include "DependencyParser.h"
#include "GranuleMapper.h"
#include "GridLayerDescriptor.h"
#include "NodeDescriptor.h"
#include "NodeSet.h"
#include "GranuleSet.h"
#include "Graph.h"
#include "Phase.h"
#include <unistd.h>
//#include <sys/socket.h>
#include "CommunicationEngine.h"
#ifdef USING_BLUEGENE
#include "BG_AvailableMemory.h"
#include <spi/include/kernel/memory.h>
#endif


#include <string>
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits.h>

#ifndef DISABLE_PTHREADS

#include "ThreadPool.h"
#include "GraphicalUserInterface.h"


#define LENS_PT_LOCK(x) pthread_mutex_lock(&x)
#define LENS_PT_UNLOCK(x) pthread_mutex_unlock(&x)

#else

#define LENS_PT_LOCK(x)
#define LENS_PT_UNLOCK(x)

#endif // DISABLE_PTHREADS

#if defined(HAVE_GPU) 
   #include <cuda_runtime_api.h>
#endif
#include "rndm.h"


// Set states; _iteration to 1
#ifndef DISABLE_PTHREADS
//Simulation::Simulation(int N, bool bindThreadsToCpus, int numWorkUnits, unsigned seed)
Simulation::Simulation(int N, bool bindThreadsToCpus, int numWorkUnits, unsigned seed, 
    int gpuID)
#else // DISABLE_PTHREADS
//Simulation::Simulation(int numWorkUnits, unsigned seed)
Simulation::Simulation(int numWorkUnits, unsigned seed, int gpuID)
#endif // DISABLE_PTHREADS

   :  _state(Simulation::_UNUSED), _iteration(0), _ntm(0), _etm(0), _root(0),
      _triggerRegistry(0), _publisherRegistry(0), _granuleMapperRegistry(0), _functorRegistry(0),
      _structRegistry(0), _constantRegistry(0), _variableRegistry(0), _ui(0),
      _triggeredPauseAction(0),

#ifndef DISABLE_PTHREADS
      _numThreads(N), _threadPool(0), _numCpus(0),
#endif // DISABLE_PTHREADS

      _pauserStatus(false), _pauser(0), _stopper(0), _erd(true),

      _passType(_GRANULE_MAPPER_PASS), _granuleMapperCount(0), _variableGranuleMapperIndex(UINT_MAX),
      _variableGlobalId(0), _variableGranuleMapperAlreadyAdded(false), _globalGranuleId(0),
      _graphSize(0), _graph(0), _publisher(0), _dependencyParser(0), _detachUserInterface(false)
#ifdef  HAVE_MPI
      , _iSenders(0), _iReceivers(0), _commEngine(0), _P2P(false), _alltoallw(false), _alltoallv(true)
#endif  //HAVE_MPI
      , _numWorkUnits(numWorkUnits), _numGranules(0), _partitioner(0)
{
   std::string fileName;
   fileName = "/so/Dependfile";

#ifndef DISABLE_PTHREADS
   pthread_mutex_init(&_stateMutex, NULL);
   pthread_mutex_init(&_timerMutex, NULL);
   pthread_mutex_init(&_socketsMutex, NULL);

   // First determine the number of CPUS online
   _numCpus = sysconf(_SC_NPROCESSORS_ONLN);
   if (_numThreads == 0) _numThreads = _numCpus;

   if (_numThreads != 1) {
      //Create a thread pool with the desired number of threads
      _threadPool = new ThreadPool(_numThreads, _numCpus, bindThreadsToCpus);
   }
#endif // DISABLE_PTHREADS

   _root = new Repertoire("Root");
   _publisher = new SimulationPublisher(*this);
   _pauser = new Pauser(*this);
   _stopper = new Stopper(*this);
   _triggerRegistry = new TypeRegistry<TriggerType>();
   _publisherRegistry = new PublisherRegistry(*this);
   _granuleMapperRegistry = new TypeRegistry<GranuleMapperType>();
   _functorRegistry = new TypeRegistry<FunctorType>();
   _structRegistry = new TypeRegistry<StructType>();
   _constantRegistry = new TypeRegistry<ConstantType>();
   _variableRegistry = new TypeRegistry<VariableType>();
   _instanceFactoryRegistries.push_back(_triggerRegistry);
   _instanceFactoryRegistries.push_back(_functorRegistry);
   _instanceFactoryRegistries.push_back(_structRegistry);
   _triggeredPauseAction = new TriggeredPauseAction();
   _dependencyParser = new DependencyParser(fileName);
   _ntm = new TypeManager<NodeType>();
   _etm = new TypeManager<EdgeType>();

   LENS_PT_LOCK(_timerMutex);
   _simTimer.start();
   if (_rank==0) printf("Simulation construct start: t = %lf\n\n", _simTimer.lapWallTime());
   LENS_PT_UNLOCK(_timerMutex);

#ifdef HAVE_MPI
   _phaseNames.push_back("FLUSH_LENS");
   _communicatingPhases["FLUSH_LENS"]=true;
   MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &_nump);
#else
   _rank=0;
   _nump=1;
#endif // HAVE_MPI

#if defined(HAVE_GPU) 
   int deviceCount = -1; // number of devices
   int dev = 0;

//#define DEBUG_FLATARRAY
#ifdef DEBUG_FLATARRAY
#else
   gpuErrorCheck(cudaGetDeviceCount(&deviceCount));

   if (deviceCount == 0) {
     fprintf(stderr, "No CUDA devices found\n");
     return;
   }
   else{
     std::cout << "There are " << deviceCount << " CUDA devices found\n";
   }


   if (gpuID < 0)
   {
     dev = _rank % int(deviceCount);
   }else {
     dev = gpuID;
   }
   cudaError_t error = cudaSetDevice(dev);
   if (error != cudaSuccess) {
     fprintf(stderr, "Error setting device to %d on rank %d (%d processes): %s\n",
	 dev, _rank, _nump, cudaGetErrorString(error));
     return;
   }
   else{
     printf("     rank %d get GPU %d\n", _rank, dev);
     print_GPU_info(dev);
   }
#endif
#endif
 
   // Seed the random number generator
   _rng.reSeed(seed, _rank) ;
   _rngShared.reSeedShared(seed-1);
   _rngSeed=seed;
   LENS_PT_LOCK(_timerMutex);
   if (_rank==0) printf("Simulation construct end: t = %lf\n\n", _simTimer.lapWallTime());
   _simTimer.reset();
   LENS_PT_UNLOCK(_timerMutex);
}

#if defined(HAVE_GPU) 
void Simulation::print_GPU_info(int devID)
{
  cudaDeviceProp prop;
  gpuErrorCheck(cudaGetDeviceProperties(&prop, devID));
  printf("Device Number: %d\n", devID);
  printf("  Device name: %s\n", prop.name);
  printf("  CC: %d.%d\n", prop.major, prop.minor);
  printf("  Memory Clock Rate (KHz): %d\n",
      prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n",
      prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
      2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}
#endif

void Simulation::pauseHandler()
{
   //now perform all the browsing operations
   _triggeredPauseAction->startAction();

   LENS_PT_LOCK(_stateMutex);
   if (_state != _STOP) {
      _state = _RUN;
   }
   LENS_PT_UNLOCK(_stateMutex);
   return;
}

void Simulation::updateAll()
{
  runPhases(_runtimePhases);
}

void Simulation::resumeHandler()
{
   // As long as user doesnt type stop or pause, keep looping
   while (_state == _RUN) {
      _iteration++;
#if 0 //def USING_BLUEGENE
      if (_iteration%1000==0) {
	//double localMemory=AvailableMemory();
	uint64_t avail;
	Kernel_GetMemorySize(KERNEL_MEMSIZE_HEAPAVAIL, &avail);
	double localMemory = ((double) avail) / ((double) 1024*1024);
	double globalMinMemory;
	MPI_Allreduce(&localMemory, &globalMinMemory, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
	if (_rank==0) {
	  std::cout<<"On iteration "<<_iteration<<", min rank memory is "<<localMemory<<"."<<std::endl;
	}
      }
#endif
      updateAll();
   }
   return;
}

void Simulation::run()
{

   LENS_PT_LOCK(_stateMutex);
   _state = _RUN;
   LENS_PT_UNLOCK(_stateMutex);

}

bool Simulation::start()
{
   LENS_PT_LOCK(_timerMutex);
   _simTimer.start();
   if (_rank==0) printf("Initialization start: t = %lf\n\n", _simTimer.lapWallTime());
   LENS_PT_UNLOCK(_timerMutex);
#ifdef HAVE_MPI
   // set up the communication infrastructure
   std::vector<int> pIdList;
   std::list<DistributableCompCategoryBase*>::iterator ccit,ccend;
   ccend = _distCatList.end();
   for(ccit = _distCatList.begin(); ccit != ccend; ++ccit) {
     (*ccit)->resetSendProcessIdIterators();
     while (!(*ccit)->atSendProcessIdEnd()) {
       pIdList.push_back((*ccit)->getSendNextProcessId());
     }
   }
   std::sort(pIdList.begin(), pIdList.end(), std::less<int>());
   std::vector<int>::iterator new_end=unique(pIdList.begin(), pIdList.end());
   pIdList.erase(new_end, pIdList.end());

   std::vector<int>::iterator pIdListIter=pIdList.begin();
   _iSenders = new IIterator <ISender> (pIdList.size());
   int i = 0;
   for (ISender* is = _iSenders->getFirst(); is != NULL; is = _iSenders->getNext()) {
      _pidsVsOrders[*pIdListIter] = i++;
      OutputStream* os = new MpiOutputStream(*pIdListIter, P2P_TAG, this);
      _outputStreams.push_back(os);
      is->setSimulationPtr(this);
      is->setRank(*pIdListIter++);
   }

   pIdList.clear();
   for(ccit = _distCatList.begin(); ccit != ccend; ++ccit) {
     (*ccit)->resetReceiveProcessIdIterators();
     while (!(*ccit)->atReceiveProcessIdEnd()) {
       pIdList.push_back((*ccit)->getReceiveNextProcessId());
     }
     std::map<std::string, bool>::const_iterator pctit, pctend=(*ccit)->getPhaseCommunicationTable().end();
     for (pctit=(*ccit)->getPhaseCommunicationTable().begin(); pctit!=pctend; ++pctit) {
       std::string simPhase=(*ccit)->getSimulationPhaseName(pctit->first);
       std::map<std::string, bool>::iterator cpiter=_communicatingPhases.find(simPhase);
       assert(cpiter!=_communicatingPhases.end());
       cpiter->second = cpiter->second || pctit->second;
     }
   }
   sort(pIdList.begin(), pIdList.end(), std::less<int>());
   new_end=unique(pIdList.begin(), pIdList.end());
   pIdList.erase(new_end, pIdList.end());

   pIdListIter=pIdList.begin();
   _iReceivers = new IIterator <IReceiver> (pIdList.size());
   for (IReceiver* ir = _iReceivers->getFirst(); ir != NULL; ir = _iReceivers->getNext()) {
      ir->initialize(this, *pIdListIter++);
   }

   _commEngine = new CommunicationEngine(_nump, _iSenders, _iReceivers, this);

#endif

   //this is so that the user can browse with the SBrowser (which
   // is a pauseActionable, and is only activated when the simulation
   // is paused) before the simulation is started.
   // DHC 8/03

   int numberOfCores = 1;

#ifndef DISABLE_PTHREADS

   numberOfCores = _numWorkUnits;

   if (_ui == 0) { // no mutex needed because there is no user interface.
      // We have to start the simulation here, otherwise the browser
      // would start it.
      run();
   } else {
      GraphicalUserInterface* g = dynamic_cast<GraphicalUserInterface*>(_ui);

      if (g) {
	 _triggeredPauseAction->startAction();
      }

      int simstate;

      do {
	 // added mutex here because it wasn't always reading a changed state
	 //when the server invoked the run() method from the UI thread
	 // -DC 7/03
	 LENS_PT_LOCK(_stateMutex);
	 simstate = _state;
	 LENS_PT_UNLOCK(_stateMutex);

      } while(simstate== _UNUSED);

      if (simstate==_STOP) {
	 return _detachUserInterface;
      }
   }

#else
      run();

#endif // DISABLE_PTHREADS

   // call optimize now on all CompCategories in case there is
   // any computational optimization waiting

   if (_distCatList.size() > 0) {

      if (_rank==0) printf("Initializing Comp Category Partitions.\n\n");

      std::list<DistributableCompCategoryBase*>::iterator it, end;
      end = _distCatList.end();
      for(it = _distCatList.begin(); it != end; ++it) {
	 (*it)->initPartitions(numberOfCores, 1);
      }
   }

   if (_edgeCatList.size() > 0) {
      std::list<EdgeCompCategoryBase*>::iterator it, end;
      end = _edgeCatList.end();
      for(it = _edgeCatList.begin(); it != end; ++it) {
	(*it)->initPartitions(numberOfCores, 0);
      }
   }

   if (_initPhases.size() > 0) {
      if (_rank==0) printf("Running Init Phases.\n\n");
      runPhases(_initPhases);
   }

#ifdef HAVE_MPI
   if (_rank==0) printf("Flushing Proxies.\n\n");
   _phaseName =  "FLUSH_LENS";
   if (_communicatingPhases[_phaseName]) {
     bool rebuildRequested = _commEngine->Communicate();
     unsigned rebuild = rebuildRequested ? 1 : 0, recommunicate=0;
     MPI_Allreduce(&rebuild, &recommunicate, 1, MPI_UNSIGNED, MPI_LOR, MPI_COMM_WORLD);
     if (recommunicate) {
       delete _commEngine;
       _commEngine = new CommunicationEngine(_nump, _iSenders, _iReceivers, this);
       rebuildRequested = _commEngine->Communicate();
       assert(!rebuildRequested);
     }
     if (_P2P) MPI_Barrier(MPI_COMM_WORLD);
   }
#endif

   if (_rank==0) printf("Starting Simulation.\n\n");
   LENS_PT_LOCK(_timerMutex);

   if (_rank==0) printf("Initialization complete : t = %lf\n\n", _simTimer.lapWallTime());
   _simTimer.reset();
   _simTimer.start();
   LENS_PT_UNLOCK(_timerMutex);
   _mark = 0;
   if (_rank==0) printf("Mark begin : t = 0\n\n");

   LENS_PT_LOCK(_stateMutex);
   _state = _RUN;
   LENS_PT_UNLOCK(_stateMutex);

   while (_state != _TERMINATE) {
      switch(_state) {
      case (_PAUSE):
	 pauseHandler();
	 break;
      case (_STOP):
	 stopHandler();
	 break;
      case (_RUN):
	 resumeHandler();
	 break;
      case (_UNUSED):
      default:
	 if (_rank==0) std::cerr << "Invalid state in Simulation::start!" << std::endl;
      }
   }
   return _detachUserInterface;
}

void Simulation::pause()
{
   if (_ui == 0) {
      if (_rank==0) std::cout << "There is no user interface, the user can't resume, stopping instead of pausing." << std::endl;
      stop();
      return;
   }

   LENS_PT_LOCK(_stateMutex);
   _state = _PAUSE;
   LENS_PT_UNLOCK(_stateMutex);
   LENS_PT_LOCK(_timerMutex);
   if (_rank==0) std::cout << std::endl << "Mark pause : t + " << _simTimer.lapWallTime() - _mark
	     << std::endl << std::endl;
   LENS_PT_UNLOCK(_timerMutex);
}

void Simulation::resume()
{
   LENS_PT_LOCK(_timerMutex);
   _mark = _simTimer.lapWallTime();
   LENS_PT_UNLOCK(_timerMutex);

   LENS_PT_LOCK(_stateMutex);
   if (_rank==0) std::cout << "Resuming simulation." << std::endl << std::endl << "Mark resume : t = " << _mark << std::endl << std::endl;
   _state = _RUN;
   LENS_PT_UNLOCK(_stateMutex);
}

void Simulation::stop()
{
   LENS_PT_LOCK(_stateMutex);
   _state = _STOP;
   LENS_PT_UNLOCK(_stateMutex);
}

void Simulation::stopHandler()
{
   LENS_PT_LOCK(_socketsMutex);
   if (_socketsInUse.size() > 0) {
      if (_rank==0) std::cerr << "Closing sockets." << std::endl;
      std::list<int>::iterator it, end = _socketsInUse.end();
      for (it = _socketsInUse.begin(); it != end; ++it) {
	 close(*it);
      }
   }
   LENS_PT_UNLOCK(_socketsMutex);
   LENS_PT_LOCK(_timerMutex);
   if (_rank==0) {
     printf("\nSimulation ends on iteration %d.\n", _iteration);
     printf("\nMark end : t = %lf\n\n", _simTimer.lapWallTime());
   }
   _simTimer.stop();
   LENS_PT_UNLOCK(_timerMutex);

   if (_finalPhases.size() > 0) {
      if (_rank==0) printf("Running Final Phases.\n\n");
      runPhases(_finalPhases);
   }

   LENS_PT_LOCK(_stateMutex);
   _state = _TERMINATE;
   LENS_PT_UNLOCK(_stateMutex);
}

void Simulation::registerDistCompCat(DistributableCompCategoryBase* c)
{
   _distCatList.push_front(c);
}

void Simulation::registerCompCat(CompCategory* c)
{
   _catList.push_front(c);
}

void Simulation::registerEdgeCompCat(EdgeCompCategoryBase* c)
{
   _edgeCatList.push_front(c);
}

float Simulation::getTime()
{
   LENS_PT_LOCK(_timerMutex);
   float rval = _simTimer.lapWallTime();
   LENS_PT_UNLOCK(_timerMutex);
   return rval;
}

void Simulation::addSocket(int fd)
{
   LENS_PT_LOCK(_socketsMutex);
   _socketsInUse.push_back(fd);
   LENS_PT_UNLOCK(_socketsMutex);
}

Simulation::~Simulation()
{
   // May need to move this somewhere else.
//   delete _ntm;
   delete _functorRegistry;
   delete _triggerRegistry;
   delete _publisherRegistry;
   delete _structRegistry;
   delete _granuleMapperRegistry;
   delete _constantRegistry;
   delete _variableRegistry;
   delete _triggeredPauseAction;
   delete _root;
   delete _publisher;
   delete _dependencyParser;
   delete _pauser;
   delete _stopper;
   delete _ntm;
   delete _etm;
   delete _graph;
#ifdef HAVE_MPI
   delete _commEngine;
#endif
   std::list<int>::iterator socketIt, socketEnd = _socketsInUse.end();
   for (socketIt = _socketsInUse.begin(); socketIt != socketEnd; ++socketIt) {
      close(*socketIt);
   }
   _socketsInUse.clear();

   std::list<SeparationConstraint*>::iterator sIt,
      sEnd = _separationConstraints.end();

   for (sIt = _separationConstraints.begin(); sIt != sEnd; ++sIt) {
      delete *sIt;
   }

#ifndef DISABLE_PTHREADS

   delete _threadPool;

#endif // DISABLE_PTHREADS

#ifdef HAVE_MPI
   if (_P2P) MPI_Barrier(MPI_COMM_WORLD);
   std::vector<OutputStream*>::iterator it2, end2 = _outputStreams.end();
   for (it2 = _outputStreams.begin(); it2 != end2; ++it2) {
      delete *it2;
   }
   MPI_Barrier(MPI_COMM_WORLD);
#endif // HAVE_MPI

   if (_rank==0) printf("Simulation destructed.\n");
}

void Simulation::addInitPhase(const std::string& name, machineType mType)
{
   PhaseElement phase(name, mType);
   _initPhases.push_back(phase);
   _phaseNames.push_back(name);
   _machineTypes[name]=mType;
   #ifdef HAVE_MPI
   _communicatingPhases[name]=false;
   #endif
}

void Simulation::addRuntimePhase(const std::string& name, machineType mType)
{
   PhaseElement phase(name, mType);
   _runtimePhases.push_back(phase);
   _phaseNames.push_back(name);
   _machineTypes[name]=mType;
   #ifdef HAVE_MPI
   _communicatingPhases[name]=false;
   #endif
}

void Simulation::addLoadPhase(const std::string& name, machineType mType)
{
   PhaseElement phase(name, mType);
   _loadPhases.push_back(phase);
   _phaseNames.push_back(name);
   _machineTypes[name]=mType;
   #ifdef HAVE_MPI
   _communicatingPhases[name]=false;
   #endif
}

void Simulation::addFinalPhase(const std::string& name, machineType mType)
{
   PhaseElement phase(name, mType);
   _finalPhases.push_back(phase);
   _phaseNames.push_back(name);
   _machineTypes[name]=mType;
   #ifdef HAVE_MPI
   _communicatingPhases[name]=false;
   #endif
}

PhaseElement& Simulation::getPhaseElement(const std::string& name)
{
   std::deque<PhaseElement>::iterator it, end;
   end = _initPhases.end();
   for (it = _initPhases.begin(); it != end; ++it) {
      if (it->getName() == name) {
	 return *it;
      }
   }
   end = _runtimePhases.end();
   for (it = _runtimePhases.begin(); it != end; ++it) {
      if (it->getName() == name) {
	 return *it;
      }
   }
   end = _loadPhases.end();
   for (it = _loadPhases.begin(); it != end; ++it) {
      if (it->getName() == name) {
	 return *it;
      }
   }
   end = _finalPhases.end();
   for (it = _finalPhases.begin(); it != end; ++it) {
      if (it->getName() == name) {
	 return *it;
      }
   }
   // should not reach here.
   if (_rank==0) std::cerr << name << " is not found as a valid phase." << std::endl;
   exit(-1);
   return *(_initPhases.begin());
}

void Simulation::addWorkUnits(const std::string& name,
			      std::deque<WorkUnit*>& workUnits)
{
   std::deque<WorkUnit*>& pDeque = getPhaseElement(name).getWorkUnits();
   pDeque.insert(pDeque.end(), workUnits.begin(), workUnits.end());
}

void Simulation::addTrigger(const std::string& name, Trigger* trigger)
{
   getPhaseElement(name).getTriggers().push_back(trigger);
}

void Simulation::runPhases(std::deque<PhaseElement>& phases)
{
   std::deque<PhaseElement>::iterator it, end = phases.end();
   std::deque<Trigger*>::iterator it2, end2;


   // If threads are disabled, or even though threads are enabled
   // if there is only one thread; kernel will process serially otherwise
   // it will be multi-threaded.

#ifndef DISABLE_PTHREADS

   if (_numThreads == 1) { // The following is single-threaded

#endif // DISABLE_PTHREADS


      for(it = phases.begin(); it != end; ++it) {

         _phaseName = it->getName();
	 end2 = it->getTriggers().end();
	 for(it2 = it->getTriggers().begin(); it2 != end2; ++it2) {
	    (*it2)->conditionalFire();
	 }
	 std::deque<WorkUnit*>::iterator it3, end3 = it->getWorkUnits().end();
	 for(it3 = it->getWorkUnits().begin(); it3 != end3; ++it3) {
	    (*it3)->execute();
	 }
#if defined(HAVE_GPU) 
	 //TUAN TODO: consider sync based on stream later
	 cudaDeviceSynchronize();
#endif

#ifdef HAVE_MPI
	 if (_communicatingPhases[_phaseName]) {
	   bool rebuildRequested = _commEngine->Communicate();
	   if (&phases == &_initPhases) {
	     unsigned rebuild = rebuildRequested ? 1 : 0, recommunicate=0;
	     MPI_Allreduce(&rebuild, &recommunicate, 1, MPI_UNSIGNED, MPI_LOR, MPI_COMM_WORLD);
	     if (recommunicate) {
	       delete _commEngine;
	       _commEngine = new CommunicationEngine(_nump, _iSenders, _iReceivers, this);
	       rebuildRequested = _commEngine->Communicate();
	       assert(!rebuildRequested);
	     }
	   }	
	   if (_P2P) MPI_Barrier(MPI_COMM_WORLD);
	 }
	 //if (&phases == &_initPhases && it==phases.begin()) while(1) {}
#endif
      }

#ifndef DISABLE_PTHREADS

   } else { // Multi-threaded
      for(it = phases.begin(); it != end; ++it) {
         _phaseName = it->getName();
	 std::deque<WorkUnit*>::iterator wuEnd = it->getWorkUnits().end();
	 end2 = it->getTriggers().end();
	 for(it2 = it->getTriggers().begin(); it2 != end2; ++it2) {
	    if ((*it2)->status()) {
	       (*it2)->fireSerial();
	       std::deque<WorkUnit*>& workUnits = (*it2)->getWorkUnits();
	       if (workUnits.size() > 0) {
		  // don't use wuEnd
		  it->getWorkUnits().insert(
		     it->getWorkUnits().end(), workUnits.begin(),
		     workUnits.end());
	       }
	    }
	 }

	 // if there is work to do
	 if (it->getWorkUnits().size() > 0) {
	    _threadPool->processQueue(it->getWorkUnits());
	 }

	 if (wuEnd != it->getWorkUnits().end()) {
	    it->getWorkUnits().erase(wuEnd, it->getWorkUnits().end());
	 }
#ifdef HAVE_MPI
	 if (_communicatingPhases[_phaseName]) {
	   bool rebuildRequested = _commEngine->Communicate();
	   if (&phases == &_initPhases) {
	     unsigned rebuild = rebuildRequested ? 1 : 0, recommunicate=0;
	     MPI_Allreduce(&rebuild, &recommunicate, 1, MPI_UNSIGNED, MPI_LOR, MPI_COMM_WORLD);
	     if (recommunicate) {
	       delete _commEngine;
	       _commEngine = new CommunicationEngine(_nump, _iSenders, _iReceivers, this);
	       rebuildRequested = _commEngine->Communicate();
	       assert(!rebuildRequested);
	     }
	   }
	   if (_P2P) MPI_Barrier(MPI_COMM_WORLD);
	 }
#endif
     }
   }
#endif // DISABLE_PTHREADS
}

std::string Simulation::findLaterPhase(const std::string& first,
				       const std::string& second)
{
   bool firstAppeared = false, secondAppeared = false;
   std::deque<PhaseElement>::iterator it, end;
   end = _initPhases.end();
   for (it = _initPhases.begin(); it != end; ++it) {
      if (it->getName() == first) {
	 if (secondAppeared) {
	    return first;
	 } else {
	    return second;
	 }
      }
      if (it->getName() == second) {
	 if (firstAppeared) {
	    return second;
	 } else {
	    return first;
	 }
      }
   }
   end = _runtimePhases.end();
   for (it = _runtimePhases.begin(); it != end; ++it) {
      if (it->getName() == first) {
	 if (secondAppeared) {
	    return first;
	 } else {
	    return second;
	 }
      }
      if (it->getName() == second) {
	 if (firstAppeared) {
	    return second;
	 } else {
	    return first;
	 }
      }
   }
   end = _loadPhases.end();
   for (it = _loadPhases.begin(); it != end; ++it) {
      if (it->getName() == first) {
	 if (secondAppeared) {
	    return first;
	 } else {
	    return second;
	 }
      }
      if (it->getName() == second) {
	 if (firstAppeared) {
	    return second;
	 } else {
	    return first;
	 }
      }
   }
   end = _finalPhases.end();
   for (it = _finalPhases.begin(); it != end; ++it) {
      if (it->getName() == first) {
	 if (secondAppeared) {
	    return first;
	 } else {
	    return second;
	 }
      }
      if (it->getName() == second) {
	 if (firstAppeared) {
	    return second;
	 } else {
	    return first;
	 }
      }
   }
   // should not reach here.
   if (_rank==0) std::cerr << "Can not find the later phase of: " << first << " "
	     << second << std::endl;
   exit(-1);
   return first;
}

machineType Simulation::getPhaseMachineType(std::string const & name) {
  std::map<std::string, machineType>::iterator
    mtiter=_machineTypes.find(name);
  assert(mtiter!=_machineTypes.end());
  return mtiter->second;
}

std::string Simulation::getFinalRuntimePhaseName() {
   assert(_runtimePhases.size() > 0);
   return _runtimePhases[_runtimePhases.size()-1].getName();
}

const char* Simulation::getServiceName(void* data) const
{
   if (data == &(_iteration)) {
      return "Iteration";
   }
   return "Error in Service Name!";
}

const char* Simulation::getServiceDescription(void* data) const
{
   if (data == &(_iteration)) {
      return "Returns iteration number of simulation.";
   }
   return "Error in Service Description!";
}

void Simulation::resetInternals()
{
   delete _functorRegistry;
   delete _triggerRegistry;
   delete _publisherRegistry;
   //delete _granuleMapperRegistry;
   delete _structRegistry;
   delete _constantRegistry;
   delete _variableRegistry;
   delete _triggeredPauseAction;
   delete _root;
   delete _publisher;
   delete _pauser;
   delete _stopper;
   delete _ntm;
   delete _etm;

   _instanceFactoryRegistries.clear();
   _catList.clear();
   _distCatList.clear();
   _edgeCatList.clear();

   std::list<int>::iterator socketIt, socketEnd = _socketsInUse.end();
   for (socketIt = _socketsInUse.begin(); socketIt != socketEnd; ++socketIt) {
      close(*socketIt);
   }
   _socketsInUse.clear();

   _initPhases.clear();
   _runtimePhases.clear();
   _loadPhases.clear();
   _finalPhases.clear();
   _phaseNames.clear();
   _phaseNames.push_back("FLUSH_LENS");

   _root = new Repertoire("Root");
   _publisher = new SimulationPublisher(*this);
   _pauser = new Pauser(*this);
   _stopper = new Stopper(*this);
   _triggerRegistry = new TypeRegistry<TriggerType>();
   _publisherRegistry = new PublisherRegistry(*this);
   //_granuleMapperRegistry = new TypeRegistry<GranuleMapperType>();
   _functorRegistry = new TypeRegistry<FunctorType>();
   _structRegistry = new TypeRegistry<StructType>();
   _constantRegistry = new TypeRegistry<ConstantType>();
   _variableRegistry = new TypeRegistry<VariableType>();
   _instanceFactoryRegistries.push_back(_triggerRegistry);
   _instanceFactoryRegistries.push_back(_functorRegistry);
   _instanceFactoryRegistries.push_back(_structRegistry);
   _triggeredPauseAction = new TriggeredPauseAction();
   _ntm = new TypeManager<NodeType>();
   _etm = new TypeManager<EdgeType>();
   _granuleMapperCount = 0;
   _variableGranuleMapperAlreadyAdded = false;
   _variableGlobalId = 0;
}

Granule* Simulation::getGranule(const NodeDescriptor& node)
{
   unsigned gmIndex = node.getGridLayerDescriptor()->getGranuleMapperIndex();
   return _granuleMappers[gmIndex]->getGranule(node);
}

Granule* Simulation::getGranule(const unsigned granuleId)
{
   return _granuleMappers[_variableGranuleMapperIndex]->getGranule(granuleId);
}

Granule* Simulation::getGranule(const VariableDescriptor& variable)
{
  return _granuleMappers[_variableGranuleMapperIndex]->getGranule(variable);
}

void Simulation::getGranules(NodeSet& nodeSet, GranuleSet& granuleSet)
{
   const std::vector<GridLayerDescriptor*>& layers = nodeSet.getLayers();

   std::vector<GridLayerDescriptor*>::const_iterator it, end = layers.end();

   for (it = layers.begin(); it != end; ++it) {
      _granuleMappers[(*it)->getGranuleMapperIndex()]->getGranules(nodeSet, granuleSet);
   }
}

void Simulation::addGranuleMapper(std::unique_ptr<GranuleMapper>& granuleMapper)
{
   granuleMapper->setGlobalGranuleIds(_globalGranuleId);  // _globalGranuleId is incremented by this function call
   _granuleMappers.push_back(granuleMapper.release());
}

void Simulation::setGraph()
{
   _graphSize = 0;

   std::vector<Granule>::iterator gIt, gEnd = _separationGranules.end();
   for (gIt = _separationGranules.begin(); gIt != gEnd; ++gIt) {
     gIt->setGraphId(_graphSize); // this function call modifies _graphSize; graphId refers to a vertex on the graph
   }


   std::vector<GranuleMapper*>::iterator it, end = _granuleMappers.end();
   for (it = _granuleMappers.begin(); it != end; ++it) {
     (*it)->setGraphId(_graphSize); // this function call modifies _graphSize; graphId refers to a vertex on the graph
     _numGranules += (*it)->getNumberOfGranules();
   }
   for (gIt = _separationGranules.begin(); gIt != gEnd; ++gIt) {
     assert(0);
     gIt->initializeGraph(_graph);
   }

   _graph = new Graph(_graphSize, _nump);
   for (it = _granuleMappers.begin(); it != end; ++it) {
     (*it)->initializeGraph(_graph);
   }

   _graph->partition(_partitioner);
}

void Simulation::addUnseparableGranuleSet(const GranuleSet& granules)
{
   if (granules.empty()) {
      return;
   }

   if (!isCostAggregationPass()) {
      return;
   }

   std::vector<SeparationConstraint*> matches;

   std::list<SeparationConstraint*>::iterator it,
      end = _separationConstraints.end();

   for (it = _separationConstraints.begin(); it != end; ++it) {
      if ((*it)->haveCommon(granules)) {
	 matches.push_back(*it);
      }
   }

   if (matches.empty()) {
      SeparationConstraint* constraint = new SeparationConstraint();
      _separationConstraints.push_back(constraint);
      matches.push_back(constraint);
   }

   matches[0]->insertGranules(granules);

   std::vector<SeparationConstraint*>::iterator vIt, vEnd = matches.end();

   for (vIt = matches.begin() + 1; vIt != vEnd; ++vIt) {
      matches[0]->insertGranules((*vIt)->getGranules());
      delete *vIt;
      _separationConstraints.remove(*vIt);
   }
}

void Simulation::setSeparationGranules()
{
   _separationGranules.resize(_separationConstraints.size());
   std::vector<Granule>::iterator it, end = _separationGranules.end();
   std::list<SeparationConstraint*>::iterator
      sit = _separationConstraints.begin();

   for (it = _separationGranules.begin(); it != end; ++it, ++sit) {
      it->setGlobalGranuleId(_globalGranuleId++);
      GranuleSet::const_iterator gIt, gEnd = (*sit)->getGranules().end();
      for (gIt = (*sit)->getGranules().begin(); gIt != gEnd; ++gIt) {
	 (*gIt)->setDepends(&(*it));
      }
   }
}

void Simulation::benchmark_start(const std::string& msg)
{
   LENS_PT_LOCK(_timerMutex);
   _simTimer.start();
   if (_rank==0) printf("%s start: t = %lf\n\n", msg.c_str(),  _simTimer.lapWallTime());
   _prevTimeElapsed  = _simTimer.lapWallTime();
   LENS_PT_UNLOCK(_timerMutex);
}
void Simulation::benchmark_timelapsed(const std::string& msg)
{
   LENS_PT_LOCK(_timerMutex);
   if (_rank==0) printf("%s passed: t = %lf\n\n", msg.c_str(),  _simTimer.lapWallTime());
   _prevTimeElapsed  = _simTimer.lapWallTime();
   LENS_PT_UNLOCK(_timerMutex);

}
void Simulation::benchmark_set_timelapsed_diff()
{
   LENS_PT_LOCK(_timerMutex);
   _prevTimeElapsed  = _simTimer.lapWallTime();
   LENS_PT_UNLOCK(_timerMutex);
}
void Simulation::benchmark_timelapsed_diff(const std::string& msg)
{
   LENS_PT_LOCK(_timerMutex);
   if (_rank==0) printf("%s passed: t = %lf\n\n", msg.c_str(),  _simTimer.lapWallTime() - _prevTimeElapsed);
   _prevTimeElapsed  = _simTimer.lapWallTime();
   LENS_PT_UNLOCK(_timerMutex);
}
void Simulation::benchmark_end(const std::string& msg)
{
   LENS_PT_LOCK(_timerMutex);
   if (_rank==0) printf("%s end: t = %lf\n\n", msg.c_str(), _simTimer.lapWallTime());
   _simTimer.reset();
   LENS_PT_UNLOCK(_timerMutex);

}

#ifdef HAVE_MPI
OutputStream* Simulation::getOutputStream(int pid)
{
   return _outputStreams[_pidsVsOrders[pid]];
}
#endif
