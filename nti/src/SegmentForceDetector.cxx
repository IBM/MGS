// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#include "SegmentForceDetector.h"
#include "Segment.h"
#include "Decomposition.h"
#include "SegmentForce.h"
#include "TouchSpace.h"
#include "SegmentDescriptor.h"
#include "NeuronPartitioner.h"
#include "Tissue.h"
#include "SegmentForce.h"
#include "SegmentForceAggregator.h"
#include "VecPrim.h"
#include "Capsule.h"
#include "BuffFactor.h"
#include "Utilities.h"
#ifndef DISABLE_PTHREADS
#include "For.h"
#endif
#include "ThreadUserData.h"

#include <cassert>
#include <memory>

//#define SIMPLE_LJ

SegmentDescriptor SegmentForceDetector::_segmentDescriptor;
SegmentForce SegmentForceDetector::_segmentForce;

SegmentForceDetector::SegmentForceDetector(
					   const int rank,
					   const int nSlicers,
					   const int nSegmentForceDetectors,
					   const int nThreads,
					   Decomposition** decomposition,
					   TouchSpace* detectionSegmentForceSpace,
					   NeuronPartitioner* neuronPartitioner,
					   Params* params)
  : _numberOfSenders(0),
    _numberOfReceivers(0),
    _rank(rank),
    _nSlicers(nSlicers),
    _nThreads(nThreads),
    _threadUserData(0),
    _nSegmentForceDetectors(nSegmentForceDetectors),
    _decomposition(decomposition),
    _detectionSegmentForceSpace(detectionSegmentForceSpace),
    _neuronPartitioner(neuronPartitioner),
    _writeToFile(false),
    _params(params),  
    _typeInt(MPI_INT),
    _segmentData(0),
    _segmentsPerSender(0),
    _segmentDispls(0),
    _typeSegments(0),
    _segmentDataSize(0),
    _previousSegBufSize(1),
    _segmentForceArray(0),
    _segmentForceCounts(0),
    _segmentForceDispls(0),
    _segmentForceDataSize(0),
    _E0(0),
    _one(1),
    _typeDouble(MPI_DOUBLE),
    _updateCoveredSegments(false),
    _coveredSegments(0),
    _coveredSegsCount(1)
    
{
  _numberOfSenders = _numberOfReceivers = 
    (nSlicers>nSegmentForceDetectors)?nSlicers:nSegmentForceDetectors;

  SegmentForce segmentForce;
  Datatype segmentForceDatatype(3, &segmentForce);
  segmentForceDatatype.set(0, MPI_LB, 0);
  segmentForceDatatype.set(1, MPI_DOUBLE, N_SEGFORCE_DATA, segmentForce.getSegmentForceData());
  segmentForceDatatype.set(2, MPI_UB, sizeof(SegmentForce));
  _typeSegmentForce = segmentForceDatatype.commit();

#ifdef A2AW
  _typeSegments = new MPI_Datatype[_numberOfSenders]; 
#else
  _typeSegments = new MPI_Datatype[1];
#endif

  _segmentsPerSender = new int[_numberOfSenders];
  _segmentDispls = new int[_numberOfSenders];
  _segmentData = new Capsule[_previousSegBufSize];

  Capsule capsule;
#ifdef A2AW
  MPI_Aint capsuleAddress;
  MPI_Get_address(&capsule, &capsuleAddress);
  MPI_Get_address(capsule.getData(), &disp);
  disp -= capsuleAddress;
  MPI_Datatype typeCapsuleBasic;
  blocklen = N_CAP_DATA;
  MPI_Type_create_hindexed(1, &blocklen, &disp, MPI_DOUBLE, &typeCapsuleBasic);
  _typeSegments = new MPI_Datatype[_numberOfSenders];
  MPI_Datatype typeCapsule;
  MPI_Type_create_resized(typeCapsuleBasic, 0, 
			  sizeof(Capsule), &typeCapsule);
  MPI_Type_commit(&typeCapsule);
#else
  _typeSegments = new MPI_Datatype[1];
  Datatype capsuleDatatype(3, &capsule);
  capsuleDatatype.set(0, MPI_LB, 0);
  capsuleDatatype.set(1, MPI_DOUBLE, N_CAP_DATA, capsule.getData());
  capsuleDatatype.set(2, MPI_UB, sizeof(Capsule));
  _typeSegments[0] = capsuleDatatype.commit();
#endif

  for (int i=0; i<_numberOfSenders; ++i) {
#ifdef A2AW
    _typeSegments[i] = typeSegmentDataBasic;
#endif
    _segmentsPerSender[i] = 0;
    _segmentDispls[i] = 0;
  }

  _segmentForceArray = new SegmentForce[_previousSegBufSize];
  _segmentForceCounts = new int[_numberOfReceivers];
  _segmentForceDispls = new int[_numberOfReceivers];
  for (int i=0; i<_numberOfReceivers; ++i) {
    _segmentForceCounts[i] = 0;
    _segmentForceDispls[i] = 0;
  }
  _threadUserData = new ThreadUserData(_nThreads);
  for (int i=0; i<_nThreads; ++i) {
    _threadUserData->_parms[i] = new Params(*_params);
  }
}

SegmentForceDetector::~SegmentForceDetector()
{
  delete [] _typeSegments;
  delete [] _segmentsPerSender;
  delete [] _segmentDispls;
  delete [] _segmentForceArray;
  delete [] _segmentForceCounts;
  delete [] _segmentForceDispls;
  delete [] _coveredSegments;
  delete _threadUserData;
}

void SegmentForceDetector::prepareToReceive(int receiveCycle, int receivePhase, CommunicatorFunction& funPtrRef)
{
  assert(receiveCycle==0);
  switch (receivePhase) {
  case 0 :
    funPtrRef = &Communicator::allToAll; 
    break;
  case 1 :
    initializePhase1Receive();
#ifdef A2AW
    funPtrRef = &Communicator::allToAllW; 
#else
    funPtrRef = &Communicator::allToAllV; 
#endif
    break;
  default : assert(0);
  }
}

void* SegmentForceDetector::getRecvbuf(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  void* rval;
  switch (receivePhase) {
  case 0 : rval = (void*)_segmentsPerSender; break;
  case 1 : 
    if (_updateCoveredSegments) rval = (void*)_coveredSegments;
    else rval = (void*)_segmentData; 
    break;
  default : assert(0);
  }
  return rval;
}

int* SegmentForceDetector::getRecvcounts(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  int* rval;
  switch (receivePhase) {
  case 0 : rval = &_one; break;
  case 1 : rval = _segmentsPerSender; break;
  default : assert(0);
  }
  return rval;
}

int* SegmentForceDetector::getRdispls(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  int* rval;
  switch (receivePhase) {
  case 0 : assert(0); break;
  case 1 : rval = _segmentDispls; break;
  default : assert(0);
  }
  return rval;
}

MPI_Datatype* SegmentForceDetector::getRecvtypes(int receiveCycle, int receivePhase)
{
  assert(receiveCycle==0);
  MPI_Datatype* rval;
  switch (receivePhase) {
  case 0 : rval = &_typeInt; break;
  case 1 : rval = _typeSegments; break;
  default : assert(0);
  }
  return rval;
}

void SegmentForceDetector::initializePhase1Receive()
{
  int segmentsCount=0;
  for(int i=0; i<_numberOfSenders; i++) {
#ifdef A2AW
    _segmentDispls[i] = segmentsCount*sizeof(Capsule);
#else
    _segmentDispls[i] = segmentsCount;
#endif
    segmentsCount += _segmentsPerSender[i];
  }
  if (_updateCoveredSegments) {
    _coveredSegsCount = segmentsCount;
    delete [] _coveredSegments;
    _coveredSegments = new Capsule[_coveredSegsCount];
  }
  else {
    _segmentDataSize=segmentsCount;
    if (_segmentDataSize>_previousSegBufSize) {
      delete [] _segmentData;
      delete [] _segmentForceArray;
      _segmentData = new Capsule[getBuffAllocationSize(_segmentDataSize)];
      _segmentForceArray = new SegmentForce[getBuffAllocationSize(_segmentDataSize)];
      _previousSegBufSize =  getUsableBuffSize(_segmentDataSize);
    }
  }
}

void SegmentForceDetector::prepareToSend(int sendCycle, int sendPhase, CommunicatorFunction& funPtrRef)
{
  assert(sendCycle==0);
  switch (sendPhase) {
  case 0 :
    detectSegmentForces(); // if send interface isn't used, SegmentForceAnalyzer will call detectSegmentForces
    funPtrRef = &Communicator::allToAll; 
    break;
  case 1 :
    funPtrRef = &Communicator::allToAllV; 
    break;
  case 2 :
    funPtrRef = &Communicator::allReduceSum; 
    break;

  default : assert(0);
  }
}

void* SegmentForceDetector::getSendbuf(int sendCycle, int sendPhase)
{  
  assert(sendCycle==0);
  void* rval;
  switch (sendPhase) {
  case 0 : rval = (void*)_segmentForceCounts; break;
  case 1 : rval = (void*)_segmentForceArray; break;
  case 2 : rval = (void*)&_E0; break;
  default : assert(0);
  }
  return rval;
}

int* SegmentForceDetector::getSendcounts(int sendCycle, int sendPhase)
{
  assert(sendCycle==0);
  int* rval;
  switch (sendPhase) {
  case 0 : rval = &_one; break;
  case 1 : rval = _segmentForceCounts; break;
  case 2 : rval = &_one; break;
  default : assert(0);
  }
  return rval;
}
  
int* SegmentForceDetector::getSdispls(int sendCycle, int sendPhase)
{
  assert(sendCycle==0);
  int* rval;
  switch (sendPhase) {
  case 0 : assert(0); break;
  case 1 : rval = _segmentForceDispls; break;
  default : assert(0);
  }
  return rval;
}

MPI_Datatype* SegmentForceDetector::getSendtypes(int sendCycle, int sendPhase)
{
  assert(sendCycle==0);
  MPI_Datatype* rval;
  switch (sendPhase) {
  case 0 : rval = &_typeInt; break;
  case 1 : rval = &_typeSegmentForce; break;
  case 2 : rval = &_typeDouble; break;
  default : assert(0);
  }
  return rval;
}

int SegmentForceDetector::getNumberOfSendPhasesPerCycle(int sendCycle)
{
  assert(sendCycle==0);
  return SEGMENTFORCE_DETECTOR_SEND_PHASES;
}

void SegmentForceDetector::detectSegmentForces()
{
  if (_updateCoveredSegments) return;

  long int binpos = 0;
  int forceCount = 0;
  FILE *data=0;
  if(_writeToFile) {
    char filename[256];
    sprintf(filename,"outSegmentForces_SegmentForceDetector.%d",_rank);
    if((data = fopen(filename, "wb")) == NULL) {
      printf("Could not open the output file %s!\n", filename);
      MPI_Finalize();
      exit(0);
    }
    _threadUserData->_file=data;
    int t = 1;
    fwrite(&t, 4, 1, data);
    binpos = ftell(data);
    fwrite(&forceCount, 4, 1, data);
  }

  for (int i=0; i<_numberOfReceivers; ++i) {
    _segmentForceCounts[i] = 0;
    _segmentForceDispls[i] = 0;
  }

  int prevSrcRank, srcRank=-1;
  _threadUserData->resetDecompositions(*_decomposition);
  _threadUserData->resetEnergy();
#ifndef DISABLE_PTHREADS
  For<SegmentForceDetector, ThreadUserData>::execute(0, 1, _segmentDataSize, this, _threadUserData, _nThreads);  
  for(int i=0; i<_segmentDataSize; ++i) {
    Capsule* s1 = &_segmentData[i];
    Sphere& s1Sphere = s1->getSphere();
    if ((*_decomposition)->getRank(s1Sphere) != _rank) continue;
    _segmentForceArray[forceCount]=_segmentForceArray[i];
    prevSrcRank = srcRank;
    srcRank=_neuronPartitioner->getRank(s1Sphere);
    if (srcRank!=prevSrcRank) {
      _segmentForceDispls[srcRank]=forceCount;
      assert(_segmentForceCounts[srcRank]==0);
    }
    _segmentForceCounts[srcRank]++;
    forceCount++;
  }
#else
  for(int i=0; i<_segmentDataSize; ++i) {
    Capsule* s1 = &_segmentData[i];
    Sphere& s1Sphere = s1->getSphere();
    if ((*_decomposition)->getRank(s1Sphere) != _rank) continue;
    doWork(0, i, _threadUserData, 0);
    _segmentForceArray[forceCount]=_segmentForceArray[i];
    prevSrcRank = srcRank;
    srcRank=_neuronPartitioner->getRank(s1Sphere);
    if (srcRank!=prevSrcRank) {
      _segmentForceDispls[srcRank]=forceCount;
      assert(_segmentForceCounts[srcRank]==0);
    }
    _segmentForceCounts[srcRank]++;
    forceCount++;
  }
#endif
  for (int i=0; i<_nThreads; ++i) {
    _E0 += _threadUserData->_E[i];
  }

  if(_writeToFile) {
    fseek(data,binpos,SEEK_SET);
    fwrite(&forceCount, 4, 1, data);
    fclose(data);
    //printf("rank=%d, forces=%d\n",_rank,forceCount);
  }
}

void SegmentForceDetector::doWork(int threadID, int i, ThreadUserData* data, Mutex* mutex)
{
  Decomposition* decomposition = _threadUserData->_decompositions[threadID];
  Params* parms = data->_parms[threadID];
  int forceInfo[6];
  Capsule *s1=&_segmentData[i], 
    *s2=_coveredSegments,
    *s2end=_coveredSegments+_coveredSegsCount,
    *s3=_segmentData,
    *s3end=_segmentData+_segmentDataSize;

  double& E0 = data->_E[threadID];

  if (decomposition->getRank(s1->getSphere()) != _rank) return;

  ////
  // Compute Repulsion-Attraction Terms
  ////

  SegmentForce* s1Force = &_segmentForceArray[i];
  double EpsA, sigmaA;
  int typeA = _segmentDescriptor.getBranchType(s1->getKey());
  EpsA = parms->getLJEps(typeA);
  s1Force->setKey(s1->getKey());
  // Force's characteristic distance is scaled by the segment radius
  sigmaA = parms->getLJSigma(typeA);// * s1->getSphere()._radius;

  double* Fa;
  Fa = s1Force->getForces();
  Fa[0] = Fa[1] = Fa[2] = 0;
  // set up force registers
  // memcpy(Fa, s1->getEndCoordinates(), 3*sizeof(double));
  for(; s2!=s2end; ++s2) {
    if (s2->getKey()!=s1->getKey()) {
      computeForces(s1,s2, parms, Fa, E0, EpsA, sigmaA);
      if(_writeToFile) {
	assert(0);
	forceInfo[0] = _segmentDescriptor.getNeuronIndex(s1->getKey());
	forceInfo[1] = _segmentDescriptor.getBranchIndex(s1->getKey());
	forceInfo[2] = _segmentDescriptor.getSegmentIndex(s1->getKey());
	forceInfo[3] = _segmentDescriptor.getNeuronIndex(s2->getKey());
	forceInfo[4] = _segmentDescriptor.getBranchIndex(s2->getKey());
	forceInfo[5] = _segmentDescriptor.getSegmentIndex(s2->getKey());
	
#ifndef DISABLE_PTHREADS
	mutex->lock();
#endif
	fwrite(&forceInfo, 4, 6, data->_file);
	fwrite(Fa,8,3,data->_file);
#ifndef DISABLE_PTHREADS
	mutex->unlock();
#endif
      }// if (_writeToFile...
    }
  }
  for(; s3!=s3end; ++s3) {
    if (s3!=s1) {
      computeForces(s1, s3, parms, Fa, E0, EpsA, sigmaA);
      if(_writeToFile) {
	assert(0);
	forceInfo[0] = _segmentDescriptor.getNeuronIndex(s1->getKey());
	forceInfo[1] = _segmentDescriptor.getBranchIndex(s1->getKey());
	forceInfo[2] = _segmentDescriptor.getSegmentIndex(s1->getKey());
	forceInfo[3] = _segmentDescriptor.getNeuronIndex(s2->getKey());
	forceInfo[4] = _segmentDescriptor.getBranchIndex(s2->getKey());
	forceInfo[5] = _segmentDescriptor.getSegmentIndex(s2->getKey());
	
#ifndef DISABLE_PTHREADS
	mutex->lock();
#endif
	fwrite(&forceInfo, 4, 6, data->_file);
	fwrite(Fa,8,3,data->_file);
#ifndef DISABLE_PTHREADS
	mutex->unlock();
#endif
      }// if (_writeToFile...
    }
  }
}

void SegmentForceDetector::computeForces(Capsule* s1, Capsule* s2, Params* parms, double* Fa, double& E0, double EpsA, double sigmaA)
{
  Sphere& s1Sphere = s1->getSphere();
  double s1Key = s1Sphere._key;

  Sphere& s2Sphere = s2->getSphere();
  double s2Key = s2Sphere._key;

  double E;
  double F[3];
  double sumRad = s1Sphere._radius + s2Sphere._radius + parms->getRadius(s1Key) + parms->getRadius(s2Key);
  if (SqDist(s1Sphere._coords, s2Sphere._coords)<=sumRad*sumRad) {

#ifndef SIMPLE_LJ
    // Signal Interaction force between neurons

    SIParameters sip=parms->getSIParams(s1Key, s2Key);
    double EpsS = sip.Epsilon;
    if (EpsS!=0.0) {
      // Force's characteristic distance is scaled by the segment radius
      double SigmaS = sip.Sigma;// * ( s1Sphere._radius + s2Sphere._radius );
      
      F[0] = F[1] = F[2] = 0.0;
      _segmentForce.SignalInteractionForce(s1Sphere,
				      s2Sphere,
				      SigmaS,
				      EpsS,
				      E,
				      F);
      E0 += E;
      for(int ii=0;ii<3;ii++) Fa[ii] += F[ii];
    }
#endif

    // Repulsive or LJ force between Neuron segments
    int typeB = _segmentDescriptor.getBranchType(s2Key);
    double EpsB = parms->getLJEps(typeB);
    // Force's characteristic distance is scaled by the segment radius
    double eps = EpsA * EpsB;
    if (eps!=0.0) {
      double sigmaB = parms->getLJSigma(typeB);// * s2Sphere._radius;
      double sigma = 0.5 * (sigmaA + sigmaB);
      F[0] = F[1] = F[2] = 0.0;
      
#ifdef SIMPLE_LJ
      _segmentForce.LennardJonesForce(s1Sphere,
				 s2Sphere,
				 sigma,
				 eps,
				 E,
				 F);
#else
      _segmentForce.RepulsiveForce(s1Sphere,
			      s2Sphere,
			      sigma,
			      eps,
			      E,
			      F);
#endif
      E0 += E;
      for(int ii=0;ii<3;ii++) Fa[ii] += F[ii];
    }
    #ifdef VERBOSE
    printf("rank %d, segA %d, segB %d, E = %lf, LocalE = %lf, F(x,y,x) = [%lf, %lf, %lf].\n", _rank, (s1-_segmentData), (s2-_coveredSegments), E, E0, Fa[0], Fa[1], Fa[2]);
    #endif
  }
}

int SegmentForceDetector::getNumberOfReceivePhasesPerCycle(int receiveCycle)
{
  assert(receiveCycle==0);
  return SEGMENTFORCE_DETECTOR_RECEIVE_PHASES;
}
