// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "TissueContext.h"
#include "VecPrim.h"
#ifdef USING_BLUEGENE
#endif
#include <algorithm>
#include "Branch.h"

TissueContext::TissueContext()
    : _nCapsules(0),
      _capsules(0),
      _origin(0),
      _neuronPartitioner(0),
      _decomposition(0),
      _tissue(0),
      _boundarySynapseGeneratorSeed(0),
      _localSynapseGeneratorSeed(0),
      _initialized(false),
      _seeded(false),
      _rank(0),
      _mpiSize(0)
#ifdef IDEA1
      , _params(NULL)
#endif
{
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_mpiSize);
}

TissueContext::~TissueContext()
{
  if (_decomposition != _neuronPartitioner) delete _decomposition;
  delete _neuronPartitioner;
  delete[] _origin;
  delete _tissue;
}

void TissueContext::readFromFile(FILE* data, int size, int rank)
{
  // read the Decomposition data back from file
  PosType dataPos;
#ifdef BINARY64BITS
  fseeko64(data, (rank - size) * sizeof(PosType), SEEK_END);
  fread(&dataPos, sizeof(PosType), 1, data);
  fseeko64(data, dataPos, SEEK_SET);
#else
  fseek(data, (rank - size) * sizeof(PosType), SEEK_END);
  size_t s = fread(&dataPos, sizeof(PosType), 1, data);
  fseek(data, dataPos, SEEK_SET);
#endif
  s = fread(&_boundarySynapseGeneratorSeed, sizeof(long), 1, data);
  s = fread(&_localSynapseGeneratorSeed, sizeof(long), 1, data);
  s = fread(&_nCapsules, sizeof(int), 1, data);
  if (_nCapsules > 0)
  {
    _capsules = new Capsule[_nCapsules];
    int offset;
    s = fread(&offset, sizeof(int), 1, data);
    _origin = _capsules + offset;
    for (int sid = 0; sid < _nCapsules; ++sid)
    {
      _capsules[sid].readFromFile(data);
      int pass;
      s = fread(&pass, sizeof(int), 1, data);
      addToCapsuleMap(_capsules[sid].getKey(), sid, DetectionPass(pass));
    }
    int length;
    s = fread(&length, sizeof(int), 1, data);
    _touchVector.clear();
    for (int i = 0; i < length; ++i)
    {
      Touch t;
      t.readFromFile(data);
      _touchVector.push_back(t, 0);
    }
  }
}

void TissueContext::writeToFile(int size, int rank)
{
  // write the Decomposition data to file
  PosType dataPos = 0, * dataPositions = 0;
  if (rank == 0)
  {
    dataPositions = new PosType[size];
    FILE* data = fopen(_commandLine.getBinaryFileName().c_str(), "wb");
    if (data == NULL)
      std::cerr << "Warning: binary file " << _commandLine.getBinaryFileName()
                << " could not be written!" << std::endl
                << std::endl;
    else
    {
      _decomposition->writeToFile(data);
      fclose(data);
    }
  }
  int written = 0, nextToWrite = 0;
  while (nextToWrite < size)
  {
    if (nextToWrite == rank)
    {
      FILE* data = fopen(_commandLine.getBinaryFileName().c_str(), "ab");
      if (data == NULL)
        std::cerr << "Warning: binary file " << _commandLine.getBinaryFileName()
                  << " could not be written!" << std::endl
                  << std::endl;
      else
      {
#ifdef BINARY64BITS
        dataPos = ftello64(data);
#else
        dataPos = ftell(data);
#endif
        writeData(data);
        fclose(data);
      }
      written = 1;
    }
    MPI_Allreduce((void*)&written, (void*)&nextToWrite, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
  }
  MPI_Gather(&dataPos, 1, MPI_POS_TYPE, dataPositions, 1, MPI_POS_TYPE, 0,
             MPI_COMM_WORLD);
  if (rank == 0)
  {
    FILE* data = fopen(_commandLine.getBinaryFileName().c_str(), "ab");
    if (data == NULL)
      std::cerr << "Warning: binary file " << _commandLine.getBinaryFileName()
                << " could not be written!" << std::endl
                << std::endl;
    else
    {
      fwrite(dataPositions, sizeof(PosType), size, data);
      fclose(data);
    }
  }
}

void TissueContext::writeData(FILE* data)
{
  fwrite(&_boundarySynapseGeneratorSeed, sizeof(long), 1, data);
  fwrite(&_localSynapseGeneratorSeed, sizeof(long), 1, data);
  fwrite(&_nCapsules, sizeof(int), 1, data);
  if (_nCapsules > 0)
  {
    int offset = _origin - _capsules;
    fwrite(&offset, sizeof(int), 1, data);
    for (int sid = 0; sid < _nCapsules; ++sid)
    {
      _capsules[sid].writeToFile(data);
      int pass = (int)(getPass(_capsules[sid].getKey()));
      fwrite(&pass, sizeof(int), 1, data);
    }
    int length = _touchVector.getCount();
    fwrite(&length, sizeof(int), 1, data);
    TouchVector::TouchIterator titer = _touchVector.begin(),
                               tend = _touchVector.end();
    for (; titer != tend; ++titer)
    {
      titer->writeToFile(data);
    }
  }
}

// GOAL: put capsules to the capsuleMap
//
int TissueContext::setUpCapsules(int nCapsules, DetectionPass detectionPass,
                                 int rank, int maxComputeOrder)
{
  Capsule* capsEnd = _capsules + nCapsules;
  resetBranches();  // clean-up all existing data
  std::sort(_capsules, capsEnd);
  capsEnd = std::unique(_capsules, capsEnd);
  _nCapsules = capsEnd - _capsules;
  if (detectionPass != NOT_SET)
  {
    for (int sid = 0; sid < _nCapsules; ++sid)
    {
      addToCapsuleMap(_capsules[sid].getKey(), sid, detectionPass);
    }
  }
  setUpBranches(rank, maxComputeOrder);
  return _nCapsules;
}

#ifdef IDEA1
void TissueContext::makeProperComputeBranch()
{
  std::map<int, std::map<ComputeBranch*, int> > properSpanningComputeBranchSizes; // mapped by rank; proper: CBs that end in another rank, start in this rank (last 'int' holds the true size)
  std::map<int, std::map<ComputeBranch*, int> > improperSpanningComputeBranchSizes; // mapped by rank; improper: CBs that pass-through this rank, start in another rank (last 'int' holds the expected true size that will be passed to from another rank)
  int numImproperBranches = 0;
  int i = 0;
  std::map<unsigned int, std::vector<ComputeBranch*> >::iterator iiter = 
    _neurons.begin(), iend = _neurons.end();
  for (; iiter != iend; iiter++)
  {
    std::vector<ComputeBranch*>::iterator citer = iiter->second.begin(),
      cend = iiter->second.end();
    for (; citer != cend; citer++)
    {
      ComputeBranch* branch = (*citer);
      // Check for improper/proper spanning CBs
      ShallowArray<int, MAXRETURNRANKS, 100> endRanks;
      int beginRank;
      if (isProperSpanning(branch, endRanks))
        for (int n=0; n<endRanks.size(); ++n) {
          if (endRanks[n] != _rank)
          {
            assert(branch->_nCapsules > 0);
            properSpanningComputeBranchSizes[endRanks[n]][branch]=branch->_nCapsules; // to become sendbuf
          }
        }
      else if (isImproperSpanning(branch, beginRank))
      {
        assert(beginRank != _rank);
        improperSpanningComputeBranchSizes[beginRank][branch]=0; // to become recvbuf
        numImproperBranches++;
      }
      assert(! (isProperSpanning(branch, endRanks) == true and 
            isImproperSpanning(branch, beginRank) == true));

    }
  }

  // Do an MPI Collective to finish creation of _improperComputeBranchCorrectedCapsuleCountsMap
  int sendbufsize = 0;
  int* sendcounts = new int[_mpiSize];
  int* senddispls = new int[_mpiSize];
  std::map<int, std::map<ComputeBranch*, int> >::iterator miter, mend = properSpanningComputeBranchSizes.end();
  int total=0;
  for (int r=0; r<_mpiSize; ++r) {
    int count=0;
    miter = properSpanningComputeBranchSizes.find(r);
    if (miter!=mend) count = miter->second.size();
    senddispls[r]=total;
    total+=(sendcounts[r]=count);
  }
  sendbufsize = total;
  //prepare sendbuf
  int* sendbuf = new int[total];
  for (int r=0; r<_mpiSize; ++r) {
    int count=0;
    miter = properSpanningComputeBranchSizes.find(r);
    if (miter!=mend) 
    {
      std::map<ComputeBranch*,int>::iterator miter2 = miter->second.begin(),
        mend2 = miter->second.end();
      int offset =0;
      for (; miter2 != mend2; miter2++)
      {
        assert(miter2->second > 0);
        sendbuf[senddispls[r]+offset] = miter2->second;
        offset++;
      }
    }
  }

  //prepare recvbuf
  int recvbufsize = 0;
  int* recvcounts = new int[_mpiSize];
  int* recvdispls = new int[_mpiSize];
  mend = improperSpanningComputeBranchSizes.end();
  total = 0;
  for (int r=0; r<_mpiSize; ++r) {
    int count=0;
    miter = improperSpanningComputeBranchSizes.find(r);
    if (r == _rank)
      assert(miter == mend);
    if (miter!=mend) count = miter->second.size();
    recvdispls[r]=total;
    total+=(recvcounts[r]=count);
  }
  recvbufsize = total;
  int* recvbuf = new int[total];
  memset(recvbuf, 0, total* sizeof(int));

  //if (_rank == 0)
  //  std::cout << "BEFORE Send proper CB capsules info" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Alltoallv(sendbuf, sendcounts, senddispls, MPI_INT, recvbuf, recvcounts, recvdispls, MPI_INT, MPI_COMM_WORLD);
  _improperComputeBranchCorrectedCapsuleCountsMap.clear();
  int bufidx=0;
  for (int r=0; r<_mpiSize; ++r) {
    miter = improperSpanningComputeBranchSizes.find(r);
    if (miter!=mend) {
      std::map<ComputeBranch*, int>::iterator miter2=miter->second.begin(), 
        mend2=miter->second.end();
      for (; miter2!=mend2; ++miter2, ++bufidx) {
        //if (_improperComputeBranchCorrectedCapsuleCountsMap.count(miter2->first)!=0)
        //{
        //  std::cout << "---" << _improperComputeBranchCorrectedCapsuleCountsMap[miter2->first] << " and " << recvbuf[bufidx] << std::endl; 
        //}
        assert(_improperComputeBranchCorrectedCapsuleCountsMap.count(miter2->first)==0);
        _improperComputeBranchCorrectedCapsuleCountsMap[miter2->first]=recvbuf[bufidx];
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  delete [] sendbuf;
  delete [] sendcounts;
  delete [] senddispls;
  delete [] recvbuf;
  delete [] recvcounts;
  delete [] recvdispls;
  //if (_rank == 0)
  //  std::cout << "COMPLETE Send proper CB capsules info" << std::endl;
}
#endif

void TissueContext::setUpBranches(int rank, int maxComputeOrder)
{
//#ifdef IDEA1
//  std::map<int, std::map<ComputeBranch*, int> > properSpanningComputeBranchSizes; // mapped by rank; proper: CBs that end in another rank, start in this rank (last 'int' holds the true size)
//  properSpanningComputeBranchSizes.clear();
//  std::map<int, std::map<ComputeBranch*, int> > improperSpanningComputeBranchSizes; // mapped by rank; improper: CBs that pass-through this rank, start in another rank (last 'int' holds the expected true size that will be passed to from another rank)
//  improperSpanningComputeBranchSizes.clear();
//  int numImproperBranches = 0;
//#endif
  int i = 0;
  while (i < _nCapsules)
  {
    ComputeBranch* branch = new ComputeBranch;
    branch->_capsules = &_capsules[i];
    branch->_nCapsules = 1;
    key_size_t key = _capsules[i].getKey();
    unsigned int neuronIndex = _segmentDescriptor.getNeuronIndex(key);
    unsigned int branchIndex = _segmentDescriptor.getBranchIndex(key);
    unsigned int computeOrder = _segmentDescriptor.getComputeOrder(key);
    DetectionPass branchPass = getPass(key);

    if (!isSpanning(_capsules[i]) && branchPass == FIRST_PASS)
    {
      int j = i;
      while (++j < _nCapsules &&
             sameBranch(_capsules[j], neuronIndex, branchIndex, computeOrder,
                        branchPass) &&
             isConsecutiveCapsule(j))
      {
        branch->_nCapsules++;
        // Remember: computeOrder is not sufficient to detect slice boundary
        // branch termination in the case MaxComputeOrder==0
        if (isSpanning(_capsules[j])) break;
      }
    }
    i += branch->_nCapsules;

    std::map<unsigned int, std::vector<ComputeBranch*> >::iterator mapIter =
        _neurons.find(neuronIndex);
    double* coords = branch->_capsules[0].getBeginCoordinates();
    unsigned int branchOrder = _segmentDescriptor.getBranchOrder(key);
    if (mapIter != _neurons.end())
    {
      if (branchPass == FIRST_PASS)
      {
        std::vector<ComputeBranch*>& branches = mapIter->second;
        int nBranches = branches.size();
        for (int j = 0; j < nBranches; ++j)
        {
          ComputeBranch* candidateBranch = branches[j];
          if (getPass(candidateBranch->_capsules[0].getKey()) == FIRST_PASS &&
              !isOutside(candidateBranch, rank))
          {
            unsigned int candidateBranchIndex =
                _segmentDescriptor.getBranchIndex(
                    candidateBranch->_capsules[0].getKey());
            unsigned int candidateOrder = _segmentDescriptor.getBranchOrder(
                candidateBranch->_capsules[0].getKey());
            unsigned int candidateComputeOrder =
                _segmentDescriptor.getComputeOrder(
                    candidateBranch->lastCapsule().getKey());
            double* candidateCoords =
                candidateBranch->lastCapsule().getEndCoordinates();

            if ((candidateOrder == branchOrder - 1 ||
                 candidateBranchIndex == branchIndex) &&
                SqDist(coords, candidateCoords) == 0.0)
            {
              branch->_parent = candidateBranch;
              branch->_parent->_daughters.push_back(branch);
              if (computeOrder == 0 && candidateComputeOrder != maxComputeOrder)
              {
                std::cerr << "TissueContext(" << rank
                          << ") : Mismatched compute orders (1,"
                          << candidateComputeOrder << "!=" << maxComputeOrder
                          << ")!" << std::endl;
                assert(0);
              }
              if (computeOrder > 0 && candidateComputeOrder != computeOrder - 1)
              {
                std::cerr << "TissueContext(" << rank
                          << ") : Mismatched compute orders (2,"
                          << candidateComputeOrder << "!=" << computeOrder - 1
                          << ")!" << std::endl;
                assert(0);
              }
              break;
            }
          }
        }
      }
    }
    else
    {
      std::vector<ComputeBranch*> newBranchVector;
      _neurons[neuronIndex] = newBranchVector;
    }
    for (int j = 0; j < branch->_nCapsules; ++j)
    {
      branch->_capsules[j].setBranch(branch);
    }
    assert(branch->_nCapsules > 0);
    _neurons[neuronIndex].push_back(branch);
//#ifdef IDEA1
//    // Check for improper/proper spanning CBs
//    ShallowArray<int, MAXRETURNRANKS, 100> endRanks;
//    int beginRank;
//    if (isProperSpanning(branch, endRanks))
//      for (int n=0; n<endRanks.size(); ++n) {
//        if (endRanks[n] != _rank)
//        {
//          assert(branch->_nCapsules > 0);
//          properSpanningComputeBranchSizes[endRanks[n]][branch]=branch->_nCapsules; // to become sendbuf
//        }
//      }
//    else if (isImproperSpanning(branch, beginRank))
//    {
//      assert(beginRank != _rank);
//      improperSpanningComputeBranchSizes[beginRank][branch]=0; // to become recvbuf
//      numImproperBranches++;
//    }
//    assert(! (isProperSpanning(branch, endRanks) == true and 
//          isImproperSpanning(branch, beginRank) == true));
//#endif
  }

//#ifdef IDEA1
//  // Do an MPI Collective to finish creation of _improperComputeBranchCorrectedCapsuleCountsMap
//  int sendbufsize = 0;
//  int* sendcounts = new int[_mpiSize];
//  int* senddispls = new int[_mpiSize];
//  std::map<int, std::map<ComputeBranch*, int> >::iterator miter, mend = properSpanningComputeBranchSizes.end();
//  int total=0;
//  for (int r=0; r<_mpiSize; ++r) {
//    int count=0;
//    miter = properSpanningComputeBranchSizes.find(r);
//    if (miter!=mend) count = miter->second.size();
//    senddispls[r]=total;
//    total+=(sendcounts[r]=count);
//  }
//  sendbufsize = total;
//  //prepare sendbuf
//  int* sendbuf = new int[total];
//  for (int r=0; r<_mpiSize; ++r) {
//    int count=0;
//    miter = properSpanningComputeBranchSizes.find(r);
//    if (miter!=mend) 
//    {
//      std::map<ComputeBranch*,int>::iterator miter2 = miter->second.begin(),
//        mend2 = miter->second.end();
//      int offset =0;
//      for (; miter2 != mend2; miter2++)
//      {
//        assert(miter2->second > 0);
//        sendbuf[senddispls[r]+offset] = miter2->second;
//        offset++;
//      }
//    }
//  }
//
//  //prepare recvbuf
//  int recvbufsize = 0;
//  int* recvcounts = new int[_mpiSize];
//  int* recvdispls = new int[_mpiSize];
//  mend = improperSpanningComputeBranchSizes.end();
//  total = 0;
//  for (int r=0; r<_mpiSize; ++r) {
//    int count=0;
//    miter = improperSpanningComputeBranchSizes.find(r);
//    if (r == _rank)
//      assert(miter == mend);
//    if (miter!=mend) count = miter->second.size();
//    recvdispls[r]=total;
//    total+=(recvcounts[r]=count);
//  }
//  recvbufsize = total;
//  int* recvbuf = new int[total];
//  memset(recvbuf, 0, total* sizeof(int));
//
//  //if (_rank == 0)
//  //  std::cout << "BEFORE Send proper CB capsules info" << std::endl;
//
//  MPI_Barrier(MPI_COMM_WORLD);
//  MPI_Alltoallv(sendbuf, sendcounts, senddispls, MPI_INT, recvbuf, recvcounts, recvdispls, MPI_INT, MPI_COMM_WORLD);
//  _improperComputeBranchCorrectedCapsuleCountsMap.clear();
//  int bufidx=0;
//  for (int r=0; r<_mpiSize; ++r) {
//    miter = improperSpanningComputeBranchSizes.find(r);
//    if (miter!=mend) {
//      std::map<ComputeBranch*, int>::iterator miter2=miter->second.begin(), 
//        mend2=miter->second.end();
//      for (; miter2!=mend2; ++miter2, ++bufidx) {
//        //if (_improperComputeBranchCorrectedCapsuleCountsMap.count(miter2->first)!=0)
//        //{
//        //  std::cout << "---" << _improperComputeBranchCorrectedCapsuleCountsMap[miter2->first] << " and " << recvbuf[bufidx] << std::endl; 
//        //}
//        assert(_improperComputeBranchCorrectedCapsuleCountsMap.count(miter2->first)==0);
//        _improperComputeBranchCorrectedCapsuleCountsMap[miter2->first]=recvbuf[bufidx];
//      }
//    }
//  }
//  MPI_Barrier(MPI_COMM_WORLD);
//  
//  delete [] sendbuf;
//  delete [] sendcounts;
//  delete [] senddispls;
//  delete [] recvbuf;
//  delete [] recvcounts;
//  delete [] recvdispls;
//  //if (_rank == 0)
//  //  std::cout << "COMPLETE Send proper CB capsules info" << std::endl;
//#endif
}

// reset:
//    _neurons
void TissueContext::resetBranches()
{
  std::map<unsigned int, std::vector<ComputeBranch*> >::iterator mapIter,
      mapEnd = _neurons.end();
  for (mapIter = _neurons.begin(); mapIter != mapEnd; ++mapIter)
  {
    std::vector<ComputeBranch*>& branches = mapIter->second;
    for (int i = 0; i < branches.size(); ++i)
    {
      delete branches[i];
    }
    branches.clear();
  }
  _neurons.clear();
}

bool TissueContext::sameBranch(Capsule& capsule, unsigned int neuronIndex,
                               unsigned int branchIndex,
                               unsigned int computeOrder,
                               DetectionPass branchPass)
{
  return (_segmentDescriptor.getNeuronIndex(capsule.getKey()) == neuronIndex &&
          _segmentDescriptor.getBranchIndex(capsule.getKey()) == branchIndex &&
          _segmentDescriptor.getComputeOrder(capsule.getKey()) ==
              computeOrder &&
          getPass(capsule.getKey()) == branchPass);
}

bool TissueContext::isGoing(Capsule& capsule, int rank)
{
  assert(_decomposition && rank >= 0);
  Sphere endSphere;
  capsule.getEndSphere(endSphere);
  return (_decomposition->getRank(capsule.getSphere()) == rank &&
          _decomposition->getRank(endSphere) != rank);
}

bool TissueContext::isComing(Capsule& capsule, int rank)
{
  assert(_decomposition && rank >= 0);
  Sphere endSphere;
  capsule.getEndSphere(endSphere);
  return (_decomposition->getRank(capsule.getSphere()) != rank &&
          _decomposition->getRank(endSphere) == rank);
}

// check if
// the first coord of capsule is in a different volume
// from second
bool TissueContext::isSpanning(Capsule& capsule)
{
  Sphere endSphere;
  capsule.getEndSphere(endSphere);
  return (_decomposition->getRank(capsule.getSphere()) !=
          _decomposition->getRank(endSphere));
}

bool TissueContext::isOutside(ComputeBranch* branch, int rank)
{
  assert(_decomposition && rank >= 0);
  Sphere endSphere;
  branch->lastCapsule().getEndSphere(endSphere);
  return (_decomposition->getRank(branch->_capsules->getSphere()) != rank &&
          _decomposition->getRank(endSphere) != rank);
}

bool TissueContext::isConsecutiveCapsule(int index)
{
  assert(index > 0 && index < _nCapsules);
  return (
      _segmentDescriptor.getSegmentIndex(_capsules[index].getKey()) -
          _segmentDescriptor.getSegmentIndex(_capsules[index - 1].getKey()) ==
      1);
}

#ifdef IDEA1
bool TissueContext::isProperSpanning(ComputeBranch* branch, ShallowArray<int, MAXRETURNRANKS, 100>& endRanks)
{
  bool rval=false;
  if (_decomposition->getRank(branch->_capsules->getSphere())==_rank &&
      isSpanning(branch->lastCapsule())) {
    rval=true;
    float deltaR = 0;
    if (_params)
     deltaR =  _params->getRadius(branch->lastCapsule().getKey());
    _decomposition->getRanks(&branch->lastCapsule().getSphere(), branch->lastCapsule().getEndCoordinates(), deltaR, endRanks);
  }
  return rval;
}

//GOAL: return TRUE if the current CB is an improper mirror part 
//      of the CB in a different rank
bool TissueContext::isImproperSpanning(ComputeBranch* branch, int& beginRank)
{
  bool rval=false;
  //if (_decomposition->getRank(branch->lastCapsule().getSphere())!=_rank &&
  //    isSpanning(branch->lastCapsule())) {
  //  rval=true;
  //  beginRank=_decomposition->getRank(branch->lastCapsule().getSphere());
  //}
  int someRank = _decomposition->getRank(branch->_capsules->getSphere());
  if (someRank !=_rank ) {
    rval=true;
    beginRank=someRank;
  }
  return rval;
}

#endif 

unsigned int TissueContext::getRankOfBeginPoint(ComputeBranch* branch)
{
  return _decomposition->getRank(branch->_capsules[0].getSphere());
}

unsigned int TissueContext::getRankOfEndPoint(ComputeBranch* branch)
{
  Sphere endSphere;
  branch->lastCapsule().getEndSphere(endSphere);
  return _decomposition->getRank(endSphere);
}

//GOAL: check if touch is at the distal-end of the Capsule 'c'
//NOTE: it is assumed that the junction takes the last capsule only
//   However, this needs to be revised, as with '-r' option, the junction
//   may occupy more than one capsule at each side of the branchpoint
bool TissueContext::isTouchToEnd(Capsule& c, Touch& t)
{
  bool rval = false;
  rval = (c.getEndProp() <= t.getProp(c.getKey()));
  return rval;
}

bool TissueContext::isMappedTouch(Touch& t,
                                  std::map<key_size_t, int>::iterator& iter1,
                                  std::map<key_size_t, int>::iterator& iter2)
{
  key_size_t s1Key = t.getKey1();
  bool rval = ((iter1 = _firstPassCapsuleMap.find(s1Key)) !=
               _firstPassCapsuleMap.end());
  if (!rval)
    rval = ((iter1 = _secondPassCapsuleMap.find(s1Key)) !=
            _secondPassCapsuleMap.end());
  if (rval)
  {
    key_size_t s2Key = t.getKey2();
    rval = ((iter2 = _firstPassCapsuleMap.find(s2Key)) !=
            _firstPassCapsuleMap.end());
    if (!rval)
      rval = ((iter2 = _secondPassCapsuleMap.find(s2Key)) !=
              _secondPassCapsuleMap.end());
  }
  return rval;
}

// GOAL: check if this MPI process (based on the given rank 'rank')
//   should handle the given touch
#ifdef IDEA1
bool TissueContext::isLensTouch(Touch& t, int rank)
{
  bool rval = false;
  //For each touch, it has 2 capsules
  //  and this returns the index of each capsule
  //  based on the capsule's key
  std::map<key_size_t, int>::iterator iter1, iter2;
  if (isMappedTouch(t, iter1, iter2))
  {
    key_size_t s1Key = t.getKey1();
    Capsule& c1 = _capsules[iter1->second];
    int rank2HandleCapsule=-1;
    int rankOfJunction;
    CapsuleAtBranchStatus status;
    if (isPartOfExplicitJunction(c1,t, status, rankOfJunction))
    {
        rank2HandleCapsule = rankOfJunction;
    }
    else{
        rank2HandleCapsule = _decomposition->getRank(c1.getSphere());
    }
    //check if this MPI process should handle the touch
    rval = (rank == rank2HandleCapsule);
    //rval = (rank == rank2HandleCapsule) or 
    //  (rank == _decomposition->getRank(c1.getSphere()));
    if (!rval)
    {
      key_size_t s2Key = t.getKey2();
      Capsule& c2 = _capsules[iter2->second];
      if (isPartOfExplicitJunction(c2,t, status, rankOfJunction))
      {
          //rank2HandleCapsule = getJunctionMPIRank(c2);
          rank2HandleCapsule = rankOfJunction;
      }
      else{
          rank2HandleCapsule = _decomposition->getRank(c2.getSphere());
      }
      rval = (rank == rank2HandleCapsule);
      //rval = (rank == rank2HandleCapsule) or 
      //  (rank == _decomposition->getRank(c2.getSphere()));
    }
  }
  return rval;
}
#else
bool TissueContext::isLensTouch(Touch& t, int rank)
{
  bool rval = false;
  //For each touch, it has 2 capsules
  //  and this returns the index of each capsule
  //  based on the capsule's key
  std::map<key_size_t, int>::iterator iter1, iter2;
  if (isMappedTouch(t, iter1, iter2))
  {
    key_size_t s1Key = t.getKey1();
    Capsule& c1 = _capsules[iter1->second];
    //preJct== true if the pre-capsule is part of the junction
    bool preJct = (_segmentDescriptor.getFlag(s1Key) && isTouchToEnd(c1, t));
    Sphere endSphere1;
    c1.getEndSphere(endSphere1);
    //check if this MPI process should handle the touch
    rval = (preJct ? rank == _decomposition->getRank(endSphere1)
                   : rank == _decomposition->getRank(c1.getSphere()));
    if (!rval)
    {
      key_size_t s2Key = t.getKey2();
      Capsule& c2 = _capsules[iter2->second];
      bool postJct = (_segmentDescriptor.getFlag(s2Key) && isTouchToEnd(c2, t));
      Sphere endSphere2;
      c2.getEndSphere(endSphere2);
      rval = (postJct ? rank == _decomposition->getRank(endSphere2)
                      : rank == _decomposition->getRank(c2.getSphere()));
    }
  }
  return rval;
}
#endif

// PARAMS:
//   rank = MPI rank of the process calling this
void TissueContext::correctTouchKeys(int rank)
{
  SegmentDescriptor segmentDescriptor;
  TouchVector::TouchIterator tend = _touchVector.end();
  std::vector<SegmentDescriptor::SegmentKeyData> maskVector;
  maskVector.push_back(SegmentDescriptor::neuronIndex);
  maskVector.push_back(SegmentDescriptor::branchIndex);
  maskVector.push_back(SegmentDescriptor::segmentIndex);
  unsigned long long mask = _segmentDescriptor.getMask(maskVector);

  for (TouchVector::TouchIterator titer = _touchVector.begin(); titer != tend;
       ++titer)
  {
    key_size_t key1 = _segmentDescriptor.getSegmentKey(titer->getKey1(), mask);
    key_size_t key2 = _segmentDescriptor.getSegmentKey(titer->getKey2(), mask);
    std::map<key_size_t, int>::iterator mapiter,
        mapend = _firstPassCapsuleMap.end();
    bool key1Fix = false, key2Fix = false;
    for (mapiter = _firstPassCapsuleMap.begin();
         mapiter != mapend && !(key1Fix && key2Fix); ++mapiter)
    {
      key_size_t capkey = _segmentDescriptor.getSegmentKey(mapiter->first, mask);
      if (!key1Fix)
      {
        if (key1Fix = (capkey == key1)) titer->setKey1(mapiter->first);
      }
      if (!key2Fix)
      {
        if (key2Fix = (capkey == key2)) titer->setKey2(mapiter->first);
      }
    }
    mapend = _secondPassCapsuleMap.end();
    for (mapiter = _secondPassCapsuleMap.begin();
         mapiter != mapend && !(key1Fix && key2Fix); ++mapiter)
    {
      key_size_t capkey = _segmentDescriptor.getSegmentKey(mapiter->first, mask);
      if (!key1Fix)
      {
        if (key1Fix = (capkey == key1)) titer->setKey1(mapiter->first);
      }
      if (!key2Fix)
      {
        if (key2Fix = (capkey == key2)) titer->setKey2(mapiter->first);
      }
    }
    assert(key1Fix && key2Fix);
  }
}

TissueContext::DetectionPass TissueContext::addToCapsuleMap(
    key_size_t key, int index, DetectionPass detectionPass)
{
  DetectionPass rval = FIRST_PASS;
  if (detectionPass == FIRST_PASS)
  {
    _firstPassCapsuleMap[key] = index;
  }
  else
  {
    std::map<key_size_t, int>::iterator mapiter = _firstPassCapsuleMap.find(key);
    if (mapiter == _firstPassCapsuleMap.end())
    {
      _secondPassCapsuleMap[key] = index;
      rval = SECOND_PASS;
    }
    else
      (*mapiter).second = index;
  }
  return rval;
}

int TissueContext::getCapsuleIndex(key_size_t key)
{
  std::map<key_size_t, int>::iterator mapiter = _firstPassCapsuleMap.find(key);
  if (mapiter == _firstPassCapsuleMap.end())
  {
    mapiter = _secondPassCapsuleMap.find(key);
    assert(mapiter != _secondPassCapsuleMap.end());
  }
  return (*mapiter).second;
}

TissueContext::DetectionPass TissueContext::getPass(key_size_t key)
{
  DetectionPass rval = FIRST_PASS;
  if (_firstPassCapsuleMap.find(key) == _firstPassCapsuleMap.end())
  {
    rval = SECOND_PASS;
    assert(_secondPassCapsuleMap.find(key) != _secondPassCapsuleMap.end());
  }
  return rval;
}

void TissueContext::clearCapsuleMaps()
{
  _firstPassCapsuleMap.clear();
  _secondPassCapsuleMap.clear();
}

void TissueContext::seed(int rank)
{
  // use the rank of the process as the seeding number
  if (!_seeded)
  {
    RNG rangen, sharedRangen;
    rangen.reSeed(_commandLine.getSeed(), rank);
    sharedRangen.reSeedShared(_commandLine.getSeed() - 1);
    _boundarySynapseGeneratorSeed = lrandom(sharedRangen);
    _localSynapseGeneratorSeed = lrandom(rangen);
    _seeded = true;
  }
  _localSynapseGenerator.reSeed(_localSynapseGeneratorSeed, rank);
  _touchSampler.reSeed(lrandom(_localSynapseGenerator), rank);
}

void TissueContext::getCapsuleMaps(std::map<key_size_t, int>& firstPassCapsuleMap,
                                   std::map<key_size_t, int>& secondPassCapsuleMap)
{
  firstPassCapsuleMap = _firstPassCapsuleMap;
  secondPassCapsuleMap = _secondPassCapsuleMap;
}

void TissueContext::resetCapsuleMaps(
    std::map<key_size_t, int>& firstPassCapsuleMap,
    std::map<key_size_t, int>& secondPassCapsuleMap)
{
  clearCapsuleMaps();
  _firstPassCapsuleMap = firstPassCapsuleMap;
  _secondPassCapsuleMap = secondPassCapsuleMap;
}

void TissueContext::rebalance(Params* params, TouchVector* touchVector)
{
  TouchVector::TouchIterator tend = touchVector->end();

  int** localHistogram;
  key_size_t* minXYZ, *maxXYZ, *binwidth;
  int* nbinsXYZ;
  _tissue->getLocalHistogram(localHistogram, minXYZ, maxXYZ, binwidth,
                             nbinsXYZ);
  double** costHistogram = new double* [3];
  for (int d = 0; d < 3; ++d)
  {
    costHistogram[d] = new double[nbinsXYZ[d]];
    for (int i = 0; i < nbinsXYZ[d]; ++i)
    {
      costHistogram[d][i] = 0;
    }
  }

  for (int i = 0; i < _nCapsules; ++i)
  {
    if (_decomposition->getRank(_capsules[i].getSphere()) == _rank)
    {
      double* coords = _capsules[i].getBeginCoordinates();
      key_size_t key = _capsules[i].getKey();

      std::list<std::string> const* compartmentVariableTargets =
          params->getCompartmentVariableTargets(key);
      if (compartmentVariableTargets)
      {
        std::list<std::string>::const_iterator
            iter = compartmentVariableTargets->begin(),
            end = compartmentVariableTargets->end();
        for (; iter != end; ++iter)
        {
          double cost = params->getCompartmentVariableCost(*iter);
          for (int d = 0; d < 3; ++d)
          {
            int bin = int((coords[d] - minXYZ[d]) / binwidth[d]);
            if (bin >= nbinsXYZ[d]) bin = nbinsXYZ[d] - 1;
            costHistogram[d][bin] += cost;
          }
        }
      }

      std::list<Params::ChannelTarget> const* channelTargets =
          params->getChannelTargets(key);
      if (channelTargets)
      {
        std::list<Params::ChannelTarget>::const_iterator
            iter = channelTargets->begin(),
            end = channelTargets->end();
        for (; iter != end; ++iter)
        {
          double cost = params->getChannelCost((*iter)._type);
          for (int d = 0; d < 3; ++d)
          {
            int bin = int((coords[d] - minXYZ[d]) / binwidth[d]);
            if (bin >= nbinsXYZ[d]) bin = nbinsXYZ[d] - 1;
            costHistogram[d][bin] += cost;
          }
        }
      }
    }
  }
  for (int direction = 0; direction < 2; ++direction)
  {
    for (TouchVector::TouchIterator titer = touchVector->begin(); titer != tend;
         ++titer)
    {
      key_size_t key1, key2;
      if (direction == 0)
      {
        key1 = titer->getKey1();
        key2 = titer->getKey2();
      }
      else
      {
        key1 = titer->getKey2();
        key2 = titer->getKey1();
      }
      int c1Idx = getCapsuleIndex(key1);
      int c2Idx = getCapsuleIndex(key2);
      Capsule& c1 = _capsules[c1Idx];
      Capsule& c2 = _capsules[c2Idx];
      double* coords1 = c1.getBeginCoordinates();
      double* coords2 = c2.getBeginCoordinates();
      if (params->electricalSynapses() &&
          (key1 < key2 ||
           !params->symmetricElectricalSynapseTargets(key1, key2)))
      {
        std::list<Params::ElectricalSynapseTarget> const*
            electricalSynapseTargets =
                params->getElectricalSynapseTargets(key1, key2);
        if (electricalSynapseTargets)
        {
          std::list<Params::ElectricalSynapseTarget>::const_iterator
              iiter = electricalSynapseTargets->begin(),
              iend = electricalSynapseTargets->end();
          for (; iiter != iend; ++iiter)
          {
            double cost = params->getElectricalSynapseCost((*iiter)._type) *
                          (*iiter)._parameter;
            for (int d = 0; d < 3; ++d)
            {
              int bin1 = int((coords1[d] - minXYZ[d]) / binwidth[d]);
              if (bin1 >= nbinsXYZ[d]) bin1 = nbinsXYZ[d] - 1;
              int bin2 = int((coords2[d] - minXYZ[d]) / binwidth[d]);
              if (bin2 >= nbinsXYZ[d]) bin2 = nbinsXYZ[d] - 1;
              costHistogram[d][bin1] += cost;
              costHistogram[d][bin2] += cost;
            }
          }
        }
      }
      if (params->chemicalSynapses())
      {
        std::list<Params::ChemicalSynapseTarget> const* chemicalSynapseTargets =
            params->getChemicalSynapseTargets(key1, key2);
        if (chemicalSynapseTargets)
        {
          std::list<Params::ChemicalSynapseTarget>::const_iterator
              iiter = chemicalSynapseTargets->begin(),
              iend = chemicalSynapseTargets->end();
          for (; iiter != iend; ++iiter)
          {
            std::map<std::string,
                     std::pair<std::list<std::string>,
                               std::list<std::string> > >::const_iterator
                targetsIter = iiter->_targets.begin();
            double cost = 0;
            for (; targetsIter != iiter->_targets.end(); ++targetsIter)
            {
              cost += params->getChemicalSynapseCost(targetsIter->first) *
                      (*iiter)._parameter;
            }
            for (int d = 0; d < 3; ++d)
            {
              int bin = int((coords1[d] - minXYZ[d]) / binwidth[d]);
              if (bin >= nbinsXYZ[d]) bin = nbinsXYZ[d] - 1;
              costHistogram[d][bin] += cost;
            }
          }
        }
      }
    }
  }

  for (int d = 0; d < 3; ++d)
  {
    for (int i = 0; i < nbinsXYZ[d]; ++i)
    {
      int n = localHistogram[d][i] = int(SIG_HIST * costHistogram[d][i]);
    }
    delete[] costHistogram[d];
  }
  delete[] costHistogram;

  _tissue->generateAlternateHistogram();
  _decomposition->decompose();

  for (TouchVector::TouchIterator titer = touchVector->begin(); titer != tend;
       ++titer)
  {
    key_size_t key1 = titer->getKey1();
    key_size_t key2 = titer->getKey2();
    int c1Idx = getCapsuleIndex(key1);
    int c2Idx = getCapsuleIndex(key2);
    Capsule& c1 = _capsules[c1Idx];
    Capsule& c2 = _capsules[c2Idx];

    ShallowArray<int, MAXRETURNRANKS, 100> ranks1, ranks2, ranks;

    _decomposition->getRanks(&c1.getSphere(), c1.getEndCoordinates(),
                             params->getRadius(key1), ranks1);
    _decomposition->getRanks(&c2.getSphere(), c2.getEndCoordinates(),
                             params->getRadius(key2), ranks2);
    ranks = ranks1;
    ranks.merge(ranks2);

    int capRank1 = _decomposition->getRank(c1.getSphere());
    int capRank2 = _decomposition->getRank(c2.getSphere());

    ShallowArray<int, MAXRETURNRANKS, 100>::iterator ranksIter = ranks.begin(),
                                                     ranksEnd = ranks.end();
    if (ranksIter != ranksEnd)
    {
      int idx = *ranksIter;
      ++ranksIter;
      for (; ranksIter != ranksEnd; ++ranksIter)
      {
        if (idx == *ranksIter)
        {
          touchVector->mapTouch(idx, titer);
          break;
        }
        idx = *ranksIter;
      }
    }
  }
}

#ifdef IDEA1
//GOAL: check if the capsule belong to the junction or not
//   The junction has to be explicit
//NOTE: This is a better strategy
// as the previous approach assumed that the junction takes the last capsule from 
//     the parent branch only, i.e. in a branchpoint, only the last capsule of the
//     parent branch is assigned with getFlag(key_thecapsule) == 1
//   However, this needs to be revised, as with '-r' option, the junction
//   may occupy more than one capsule at each side of the branchpoint
//   status ==  1 (proximal) the capsule belong to
//              2 (distal)
//              3 (soma)
//              -1 (undefined)
//   rank = return the rank of the MPI process that handle this touch
//            (ONLY IF the touch is part of explicit junction)
bool TissueContext::isPartOfExplicitJunction(Capsule& capsule, Touch &t, int & rank, Capsule** junctionCapsule)
{
    CapsuleAtBranchStatus status;
    bool result = false;
    ComputeBranch* branch = capsule.getBranch();
    if (isPartOfExplicitJunction(capsule, t, status, rank))
    {
        result = true;
        switch (status) 
        {
            case CapsuleAtBranchStatus::PROXIMAL:
                //junctionCapsule = &(branch->_parent->lastCapsule());
                if (branch->_parent)
                    *junctionCapsule = &(branch->_parent->lastCapsule());
                else
                    *junctionCapsule = NULL;
                break;
            case CapsuleAtBranchStatus::DISTAL:
                *junctionCapsule = &(branch->lastCapsule());
                break;
            case CapsuleAtBranchStatus::SOMA:
                *junctionCapsule = &capsule;
                break;
            default:
                assert(0);
        }
    }
    return result;
}

bool TissueContext::isPartOfExplicitJunction(Capsule& capsule, Touch &t)
{ 
    CapsuleAtBranchStatus status;
    int rank;
    return isPartOfExplicitJunction(capsule, t, status, rank);
}

bool TissueContext::isPartOfExplicitJunction(Capsule& capsule, Touch &t, 
        CapsuleAtBranchStatus& status, int& rank)
{
  return isPartOfExplicitJunction(capsule, t, 
      status, rank, _decomposition);
}

bool TissueContext::isPartOfExplicitJunction(Capsule& capsule, Touch &t, 
        CapsuleAtBranchStatus& status, int& rank, Decomposition* decomposition)
{
    //NOTE: check the original implementation 'TouchDetectTissueSlicer::sliceAllNeurons()'
    //which calll Segment->isJunctionSegment(true) to configure the flag
    //TODO: remove the flag as it wont be used any more 
  ComputeBranch* branch = capsule.getBranch();
  //assert(branch->_configuredCompartment);
  int beginRank;
  bool isImproperCB = this->isImproperSpanning(branch, beginRank);
  float reserved4distend;
  float reserved4proxend;
  this->getNumCompartments(branch);
  if ( isImproperCB )
  {
    reserved4distend = branch->_numCapsulesEachSideForBranchPoint.second;
    reserved4proxend = 0;
  }
  else{
    reserved4distend = branch->_numCapsulesEachSideForBranchPoint.second;
    reserved4proxend = branch->_numCapsulesEachSideForBranchPoint.first;
    
  }

  //this->getNumCompartments(branch);
  //float reserved4distend = branch->_numCapsulesEachSideForBranchPoint.second;
  //float reserved4proxend = branch->_numCapsulesEachSideForBranchPoint.first;
  int cps_index =
      (&capsule - capsule.getBranch()->_capsules);  // zero-based index

  //# capsules in that branch
  int ncaps = branch->_nCapsules;
  int cps_index_reverse = ncaps - cps_index - 1;  // from the distal-end
  bool result = false;
  key_size_t key = capsule.getKey();
  unsigned int computeOrder =
    _segmentDescriptor.getComputeOrder(key);

  status = CapsuleAtBranchStatus::UNDEFINED;
  rank = -1; //TO ENSURE invalid value
  if (_segmentDescriptor.getBranchType(capsule.getKey()) ==
      Branch::_SOMA)  // the branch is a soma
  {
      result = true;
      status = CapsuleAtBranchStatus::SOMA;
      //rank = this->getRankOfEndPoint(capsule.getBranch());
      Sphere endSphere1;
      branch->lastCapsule().getEndSphere(endSphere1);
      rank = decomposition->getRank(endSphere1);
      //assert(rank == _decomposition->getRank(endSphere1));
  }
  else{
      if (computeOrder == 0)
      {
          //if (cps_index < reserved4proxend)
          if (cps_index < int(floor(reserved4proxend)))
              result = true;
          if (cps_index == int(floor(reserved4proxend)))
          {
              result = (t.getProp(capsule.getKey()) < (reserved4proxend-cps_index));
          }
          if (result)
          {
              status = CapsuleAtBranchStatus::PROXIMAL;
              //rank = this->getRankOfBeginPoint(capsule.getBranch());
              //assert(capsule.getBranch()->_parent);
              //rank = this->getRankOfEndPoint(capsule.getBranch()->_parent);
              rank = decomposition->getRank(branch->_capsules[0].getSphere());
              //assert(rank == _decomposition->getRank(branch->_capsules[0].getSphere()));
          }
      }
      if (!result and computeOrder == MAX_COMPUTE_ORDER and 
              reserved4distend > 0.0 and 
              branch->_daughters.size() >= 1
              /*ensure not terminal*/)
      {
          //if (cps_index_reverse < reserved4distend)
          if (cps_index_reverse < int(floor(reserved4distend)))
              result = true;
          if (cps_index_reverse == int(floor(reserved4distend)))
          {
              result = (1.0-t.getProp(capsule.getKey()) < (reserved4distend - cps_index_reverse));
          }
          if (result)
          {
              status = CapsuleAtBranchStatus::DISTAL;
              //rank = this->getRankOfEndPoint(capsule.getBranch());
              Sphere endSphere1;
              branch->lastCapsule().getEndSphere(endSphere1);
              rank = decomposition->getRank(endSphere1);
              //assert(rank == _decomposition->getRank(endSphere1));
          }
      }

  }
  //if (result)
  //    rank = getExplicitJunctionMPIRank(capsule);
  return result;
}

bool TissueContext::isPartOfExplicitJunction(Capsule& capsule, Touch &t, int& rank)
{
    CapsuleAtBranchStatus status;
    return (isPartOfExplicitJunction(capsule, t, status, rank));
}
// GOAL: for a given capsule (in the associated ComputeBranch),
// returns the index of the compartment it belongs to
//  NOTE:
//      The explicit branching junction and implicit branching junction
//      is not considered here
//  NOTE: If the capsule is at distal-end
//      check for implicit junction --> for that index should be zero
//   As the implicit branching junction has no real capsule matching
//    (we just assume ignore that capsule if it presents)
// NOTE: it handles the case of rescaling branch-point junction based on '-r' option
int TissueContext::getCptIndex(Capsule& caps, Touch & touch)
{
  int cptIndex = 0;
  ComputeBranch* branch = caps.getBranch();
  assert(branch->_configuredCompartment);
  //  std::vector<int>* cptsizes_in_branch = (_cptSizesForBranchMap[branch]);
  //if (! branch->_configuredCompartment)
  //{
  //    std::vector<int> cptsizes_in_branch;
  //    bool isDistalEndSeeImplicitBranchingPoint;
  //    int ncpts = ??? getNumCompartments(branch, cptsizes_in_branch,
  //            isDistalEndSeeImplicitBranchingPoint);
  //}
  std::vector<int>& cptsizes_in_branch = branch->_cptSizesForBranch;
  if (this->isPartOfExplicitJunction(caps, touch))
  {
    cptIndex = 0;
  }
  else{
    assert(cptsizes_in_branch.size()>0);
    int cps_index =
      (&caps - caps.getBranch()->_capsules);  // zero-based index
    //# capsules in that branch
    int ncaps = branch->_nCapsules;
    int cps_index_reverse = ncaps - cps_index - 1;  // from the distal-end
    std::vector<int>::iterator iter = cptsizes_in_branch.begin(),
      iterend = cptsizes_in_branch.end();

    int count = 0;
    for (; iter < iterend; iter++)
    {
      count = count + *iter;
      if (count >= cps_index_reverse)
      {
        break;
      }
      cptIndex++;
    }

  }
  assert(cptIndex < cptsizes_in_branch.size());
  return cptIndex;
}

int TissueContext::getCptIndex(Capsule* caps, Touch & touch)
{
    return this->getCptIndex(*caps, touch);
}

int TissueContext::getNumCompartments(ComputeBranch* branch)
{
    std::vector<int> cptsizes_in_branch;
    return this->getNumCompartments(branch, cptsizes_in_branch);
}
int TissueContext::getNumCompartments(
    ComputeBranch* branch, std::vector<int>& cptsizes_in_branch)
{
  //TUANTODO: as there are two times it is called after which the ComputeBranch
  //can change
  //we need to ensure  it still update the last 
  //if (branch->_configuredCompartment)
  //{//stop here to avoid recalculation
  //    cptsizes_in_branch = branch->_cptSizesForBranch ;
  //    return branch->_cptSizesForBranch.size();
  //}

  int rval;
  int ncpts;
  //# capsules in that branch
  int ncaps;
  
  int beginRank;
  bool isImproperCB = isImproperSpanning(branch, beginRank);
  if (isImproperCB)
  {
    assert(this->_improperComputeBranchCorrectedCapsuleCountsMap.count(branch) == 1);
    ncaps = this->_improperComputeBranchCorrectedCapsuleCountsMap[branch];
    assert(ncaps > 0);
  }
  else
   ncaps = branch->_nCapsules;

  // we need this in case the ncaps is less than _compartmentSize
  // e.g. soma has only 1 capsule
  int _compartmentSize = this->_commandLine.getCapsPerCpt();
  int cptSize = (ncaps > _compartmentSize) ? _compartmentSize : ncaps;
  // Find: # compartments in the current branch
  ncpts = (int(floor(double(ncaps) / double(cptSize))) > 0)
              ? int(floor(double(ncaps) / double(cptSize)))
              : 1;
// suppose the branch is long enough, reverse some capsules at each end for
// branchpoint
//  2. explicit slicing cut
//  3. explicit branchpoint
  cptsizes_in_branch.clear();
  Capsule* capPtr = &branch->_capsules[ncaps - 1];
  key_size_t key = capPtr->getKey();
  unsigned int computeOrder = _segmentDescriptor.getComputeOrder(key);
  float reserved4proxend = 0.0;
  float reserved4distend = 0.0;
  //NOTE: '-r' #caps/cpt
  if (ncaps == 1)
  {// -r get any value 
    cptSize = 1;
    ncpts = 1;
    //REGULAR treatment
    if (computeOrder == 0 and 
            (branch->_daughters.size() > 0 and computeOrder == MAX_COMPUTE_ORDER))
    {
        reserved4proxend = 0.25;
        reserved4distend = 0.25;
    }else if (computeOrder == 0)
    {
        reserved4proxend = 0.25;
        reserved4distend = 0.0;
    }else if (computeOrder == MAX_COMPUTE_ORDER and branch->_daughters.size() > 0
            )
    {
        reserved4proxend = 0.0;
        reserved4distend = 0.25;
    }
    else{
        reserved4proxend = 0.0;
        reserved4distend = 0.0;
    }
    //SPECIAL treatment
    if (branch->_parent)
    {
      Capsule& firstcaps = branch->_capsules[0];
      Capsule& pcaps = branch->_parent->_capsules[0];
      if (_segmentDescriptor.getBranchType(pcaps.getKey()) ==
          Branch::_SOMA)  // the parent branch is soma
      {
          float length = firstcaps.getLength();
          float somaR = pcaps.getLength(); //soma radius
          if (length <= somaR)
          {
              std::cerr << "ERROR: There is 1-capsule branch from the soma, and the point falls within the soma'radius"
                  << std::endl;
              std::cerr << " ... Please make the capsule longer\n";
              std::cerr << 
                  "Neuron index: " << _segmentDescriptor.getNeuronIndex(pcaps.getKey()) 
                  << std::endl;
              double* coord = firstcaps.getBeginCoordinates();
              std::cerr << "Coord: " << coord[0] << ", " << coord[1] << ", " << coord[2]
              << std::endl;
              assert(0);
          }
          else
          {
              reserved4proxend = somaR/length;
              reserved4distend = 0.25 * (1.0-reserved4proxend); 
          }
      }
    }
    branch->_numCapsulesEachSideForBranchPoint = std::make_pair(reserved4proxend, reserved4distend);
    cptsizes_in_branch.push_back(cptSize);
  }
  else if (ncaps == 2)
  {// -r get any value
    cptSize = 2;
    ncpts = 1;
    //REGULAR treatment
    if (computeOrder == 0 
            and (computeOrder == MAX_COMPUTE_ORDER and branch->_daughters.size() > 0))
    {
        reserved4proxend = 0.5;
        reserved4distend = 0.5;
    }else if (computeOrder == 0)
    {
        reserved4proxend = 0.5;
        reserved4distend = 0.0;
    }else if (computeOrder == MAX_COMPUTE_ORDER and branch->_daughters.size() > 0
            )
    {
        reserved4proxend = 0.0;
        reserved4distend = 0.5;
    }
    else{
        reserved4proxend = 0.0;
        reserved4distend = 0.0;
    }
    //SPECIAL treatment
    if (branch->_parent)
    {
      Capsule& firstcaps = branch->_capsules[0];
      Capsule& pcaps = branch->_parent->_capsules[0];
      if (_segmentDescriptor.getBranchType(firstcaps.getKey()) ==
          Branch::_SOMA)  // the parent branch is soma
      {//ignore the first capsule
          float length = firstcaps.getLength();
          float somaR = pcaps.getLength(); //soma radius
          if (length <= somaR)
          {//skip the first capsule 
              reserved4proxend = 1.0;
              reserved4distend = 0.25;
          }else
          {
              reserved4proxend = somaR/length;
              reserved4distend = 0.25;
          } 
      }
    }
    branch->_numCapsulesEachSideForBranchPoint = std::make_pair(reserved4proxend, reserved4distend);
    cptsizes_in_branch.push_back(cptSize);
  }
  else if (ncaps >= 3)
  {
    float fcaps_loss = 0.0;
    if (computeOrder == 0 
       and (computeOrder == MAX_COMPUTE_ORDER and branch->_daughters.size() > 0))
    {
        if (_compartmentSize >= 2)
        {
            reserved4proxend = 0.75;
            reserved4distend = 0.75;
        }else{
            reserved4proxend = 0.5;
            reserved4distend = 0.5;
        }
    }else if (computeOrder == 0)
    {
        reserved4proxend = 0.5;
        reserved4distend = 0.0;
    }else if (computeOrder == MAX_COMPUTE_ORDER and branch->_daughters.size() > 0)
    {
        reserved4proxend = 0.0;
        reserved4distend = 0.5;
    }
    else{
        reserved4proxend = 0.0;
        reserved4distend = 0.0;
    }
    //NOTE: adjust this we need to adjust "secA"
    fcaps_loss = reserved4distend + reserved4proxend; //0.75 for proximal, 0.75 for distal end

#define SMALL_FLT 0.00013
    int caps_loss_prox = (reserved4proxend < SMALL_FLT) ? 0 : 1;
    int caps_loss_dist = (reserved4distend < SMALL_FLT) ? 0 : 1;
    int ncaps_loss =  caps_loss_prox + caps_loss_dist;
    int tmpVal= int(floor(double(ncaps-ncaps_loss) / double(cptSize))); 
    ncpts = (tmpVal > 0) ? tmpVal : 1;
    cptsizes_in_branch.resize(ncpts);
    std::fill(cptsizes_in_branch.begin(), cptsizes_in_branch.end(), 0);
    //NOTE: reserve at each end
    cptsizes_in_branch[0] += caps_loss_prox; //reserve 1 for proximal
    cptsizes_in_branch[ncpts-1] += caps_loss_dist;//reserve 1 for distal
    int caps_left = ncaps - ncaps_loss;

    int count = 0;
    do{
      count++;
      for (int ii = 0; ii < ncpts; ii++)
      {
        if (caps_left > 0)
        {
          cptsizes_in_branch[ii] += 1;
          caps_left -= 1;
        }
        else
          break;
      }
      if (count == 3)
      {//every 3 capsules added to each cpt, there is 1 for prox.end branching point and 1 for dist.end branching point
        if (caps_left > 0)
        {
          if (computeOrder == MAX_COMPUTE_ORDER and branch->_daughters.size() > 0)
          {
              cptsizes_in_branch[ncpts-1] += 1;
              caps_left -= 1;
              reserved4distend += 1;
          }
        }
        if (caps_left > 0)
        {
          if (computeOrder ==  0)
          {
              cptsizes_in_branch[0] += 1;
              caps_left -= 1;
              reserved4proxend += 1;
          }
        }
        count = 0;
      }
    }while (caps_left>0);

    //SPECIAL treatment
    if (branch->_parent)
    {
        Capsule& pcaps = branch->_parent->_capsules[0];
        if (_segmentDescriptor.getBranchType(pcaps.getKey()) ==
                Branch::_SOMA)  // the parent branch is soma
        {//ignore the first capsule
            reserved4proxend = (reserved4proxend >= 1.0) ? reserved4proxend : 1.0;
        }
    }
    branch->_numCapsulesEachSideForBranchPoint = std::make_pair(reserved4proxend, reserved4distend);
  }

  branch->_cptSizesForBranch = cptsizes_in_branch;
  branch->_configuredCompartment = true;
  assert(cptsizes_in_branch.size() > 0);
  //making sure no distal-end reserve for terminal branch
  if (branch->_daughters.size() == 0)
  {
    branch->_numCapsulesEachSideForBranchPoint.second = 0.0;
  }

  //just for checking
  int  sumEle = std::accumulate(cptsizes_in_branch.begin(), cptsizes_in_branch.end(), 0);
  if (sumEle != ncaps)
  {
    std::cout << "numEle =" << sumEle << "; ncaps = " << ncaps << std::endl;
    std::vector<int>::iterator iter=cptsizes_in_branch.begin(), iterend=cptsizes_in_branch.end();
    for (iter; iter < iterend; iter++)
    {
        std::cout << (*iter) << " " ;
    }
    std::cout << std::endl;
      
  }
  assert (sumEle == ncaps);

  rval = ncpts;
  return rval;
}
#endif
