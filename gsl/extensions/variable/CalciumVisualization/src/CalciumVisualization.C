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
// =================================================================

#include "Lens.h"
#include "CalciumVisualization.h"
#include "Simulation.h"
#include "CG_CalciumVisualization.h"
#include "NeuronPartitioner.h"
#include "MaxComputeOrder.h"

#include <cfloat>
#include <cmath>
#include <memory>
#include <algorithm>
#include <arpa/inet.h>
#include <utility>
#define SWAP_BYTE_ORDER
#define N_IO_NODES 1
#define N_RECEIVE_BUFFS 2
#define N_SEND_BUFFS 2
#define BUFF_SIZE 1000

CalciumVisualization::CalciumVisualization()
    : CG_CalciumVisualization(),
      _outFile(0),
      _rank(-1),
      _size(-1),
      _isIoNode(false),
      _calciumBuffs(0),
      _nBuffs(0),
      _nSends(0),
      _nKreceives(0),
      _nVreceives(0)
{
}

void CalciumVisualization::initialize(RNG& rng)
{
  _rank = getSimulation().getRank();
  _size = getSimulation().getNumProcesses();
  assert(_size >= N_IO_NODES);
  _ioNodes.resize(N_IO_NODES);
  for (int i = 0; i < N_IO_NODES; ++i)
  {
    _ioNodes[i] = i;  // randomize this for better performance
  }
  std::vector<int>::iterator iter =
      find(_ioNodes.begin(), _ioNodes.end(), _rank);
  _isIoNode = (iter != _ioNodes.end());

  // Note: The use of new here is not thread safe!
  // Therefore, this variable object should be instantiated only once
  // per memory space, and initialized in its own InitPhase

  assert(deltaT);
  std::map<CompartmentKey, dyn_var_t*, CompartmentKey::compare> calciumMap;
  if (Ca.size() > 0)
  {
    for (unsigned int i = 0; i < Ca.size(); ++i)
    {
      key_size_t key = branchData[i]->key;
      int sid = _segmentDescriptor.getSegmentIndex(key);
      int Ca_sz = Ca[i]->size();
      for (int j = 0; j < Ca_sz; ++j)
      {
        calciumMap[CompartmentKey(key, j + sid)] = &(*Ca[i])[j];
      }
    }
    Ca.clear();
    branchData.clear();
  }

  int localDataSize = calciumMap.size();
  unsigned long keysRecvdCount = 0;

  _nSends = int(ceil(double(localDataSize) / double(BUFF_SIZE)));
  int destIO = _ioNodes[_rank % N_IO_NODES];
  for (int i = 0; i < N_IO_NODES; ++i)
  {
    int sendToDest = 0;
    if (i == _rank % N_IO_NODES) sendToDest = _nSends;
    MPI::COMM_WORLD.Reduce(&sendToDest, &_nVreceives, 1, MPI::INT, MPI::SUM,
                           _ioNodes[i]);
    MPI::COMM_WORLD.Reduce(&_nSends, &_nKreceives, 1, MPI::INT, MPI::SUM,
                           _ioNodes[i]);
  }
  _nBuffs = _isIoNode ? ((N_RECEIVE_BUFFS < _nVreceives) ? N_RECEIVE_BUFFS
                                                         : _nVreceives)
                      : ((N_SEND_BUFFS < _nSends) ? N_SEND_BUFFS : _nSends);
  if (_isIoNode) assert(_nBuffs > 0);
  key_size_t** keyBuffs = 0;

  if (_nBuffs > 0)
  {
    keyBuffs = new key_size_t* [_nBuffs];
    for (int i = 0; i < _nBuffs; ++i) keyBuffs[i] = new key_size_t[BUFF_SIZE];
    std::map<CompartmentKey, dyn_var_t*, CompartmentKey::compare>::iterator
        calciumMapIter = calciumMap.begin();
    for (int i = 0; i < localDataSize; ++i, ++calciumMapIter)
    {
      if (i % BUFF_SIZE != 0 &&
          calciumMapIter->second == _marshallPatterns.back().first + 1)
        _marshallPatterns.back().second++;
      else
        _marshallPatterns.push_back(
            std::pair<dyn_var_t*, int>(calciumMapIter->second, 1));
    }
    if (!_isIoNode)
    {
      for (int pass = 0; pass < 2; ++pass)
      {
        std::vector<std::pair<MPI::Request, key_size_t*> > requests(_nBuffs);
        std::map<key_size_t*, int> destIndices;
        for (int i = 0; i < _nBuffs; ++i)
        {
          requests[i].second = keyBuffs[i];
          destIndices[keyBuffs[i]] = 0;
        }
        std::vector<std::pair<MPI::Request, key_size_t*> >::iterator riter, rend;
        MPI::Status status;
        calciumMapIter = calciumMap.begin();
        std::map<CompartmentKey, dyn_var_t*, CompartmentKey::compare>::iterator
            calciumMapEnd = calciumMap.end();
        rend = requests.end();
        for (riter = requests.begin(); riter != rend; ++riter)
        {
          int scount;
          key_size_t* sbuff = riter->second;
          assert(destIndices[sbuff] == 0);
          for (scount = 0;
               scount < BUFF_SIZE && calciumMapIter != calciumMapEnd;
               ++scount, ++calciumMapIter)
          {
            sbuff[scount] = _segmentDescriptor.modifySegmentKey(
                SegmentDescriptor::segmentIndex, calciumMapIter->first._cptIdx,
                calciumMapIter->first._key);
          }

          riter->first = MPI::COMM_WORLD.Isend(sbuff, scount, MPI::DOUBLE,
                                               _ioNodes[destIndices[sbuff]], 0);
          if (++destIndices[sbuff] == N_IO_NODES) destIndices[sbuff] = 0;
        }
        while (calciumMapIter != calciumMapEnd)
        {
          riter = requests.begin();
          while (0 == riter->first.Test(status))
            if (++riter == rend) riter = requests.begin();
          int scount;
          key_size_t* sbuff = riter->second;
          if (destIndices[sbuff] == 0)
          {
            for (scount = 0;
                 scount < BUFF_SIZE && calciumMapIter != calciumMapEnd;
                 ++scount, ++calciumMapIter)
              sbuff[scount] = _segmentDescriptor.modifySegmentKey(
                  SegmentDescriptor::segmentIndex,
                  calciumMapIter->first._cptIdx, calciumMapIter->first._key);
          }
          riter->first = MPI::COMM_WORLD.Isend(sbuff, scount, MPI::DOUBLE,
                                               _ioNodes[destIndices[sbuff]], 0);
          if (++destIndices[sbuff] == N_IO_NODES) destIndices[sbuff] = 0;
        }
        MPI::COMM_WORLD.Barrier();
      }
    }
    else
    {  // _isIoNode
      _demarshalPatterns.resize(_size);
      int totalNeurons = 0;
      int totalSegmentsRead = 0;
      int neuronsPerLayer[6] = {0, 0, 0, 0, 0, 0};
      std::vector<int> neuronSegmentOffsets;
      NeuronPartitioner::countAllNeurons(inFileName.c_str(), totalNeurons,
                                         totalSegmentsRead, neuronsPerLayer,
                                         neuronSegmentOffsets);
      for (int i = totalNeurons - 1; i > 0; --i)
        neuronSegmentOffsets[i] = neuronSegmentOffsets[i - 1];
      neuronSegmentOffsets[0] = 0;

      for (int i = 1; i < totalNeurons; ++i)
        neuronSegmentOffsets[i] =
            neuronSegmentOffsets[i] + neuronSegmentOffsets[i - 1];

      std::vector<std::vector<int> > branchSegmentOffsets;
      branchSegmentOffsets.resize(totalNeurons);

      for (int pass = 0; pass < 2; ++pass)
      {
        keysRecvdCount = localDataSize;
        std::vector<std::pair<MPI::Request, key_size_t*> > requests(_nBuffs);
        for (int i = 0; i < _nBuffs; ++i) requests[i].second = keyBuffs[i];
        std::vector<std::pair<MPI::Request, key_size_t*> >::iterator riter;
        MPI::Status status;

        for (riter = requests.begin() + 1; riter != requests.end();
             ++riter)  // leave first for self receive
          riter->first = MPI::COMM_WORLD.Irecv(riter->second, BUFF_SIZE,
                                               MPI::DOUBLE, MPI_ANY_SOURCE, 0);

        if (pass == 1)
        {
          for (int i = 0; i < totalNeurons; ++i)
          {  // finish set up of branchSegmentOffsets
            std::vector<int>& branchOffsets = branchSegmentOffsets[i];
            int totalBranches = branchOffsets.size();
            for (int j = totalBranches - 1; j > 0; --j)
              branchOffsets[j] = branchOffsets[j - 1];
            branchOffsets[0] = 0;
            for (int j = 1; j < totalBranches; ++j)
              branchOffsets[j] = branchOffsets[j] + branchOffsets[j - 1];
          }
        }

        std::map<CompartmentKey, dyn_var_t*, CompartmentKey::compare>::iterator
            calciumMapIter = calciumMap.begin(),
            calciumMapEnd = calciumMap.end();

        int buffsRecvdCount = 0;
        while (buffsRecvdCount < _nKreceives)
        {
          int rcount, sender = _rank;
          key_size_t* rbuff = keyBuffs[0];
          if (buffsRecvdCount < _nSends)
          {  // receive from self
            for (rcount = 0;
                 rcount < BUFF_SIZE && calciumMapIter != calciumMapEnd;
                 ++rcount, ++calciumMapIter)
            {
              rbuff[rcount] = _segmentDescriptor.modifySegmentKey(
                  SegmentDescriptor::segmentIndex,
                  calciumMapIter->first._cptIdx, calciumMapIter->first._key);
            }
          }
          else
          {
            riter = requests.begin();
            while (0 == riter->first.Test(status))
              if (++riter == requests.end()) riter = requests.begin();
            sender = status.Get_source();
            rcount = status.Get_count(MPI::DOUBLE);
            rbuff = riter->second;
            keysRecvdCount += rcount;
          }

          std::vector<std::pair<int, int> > buffPattern;
          int lastPattern = -1;
          for (int i = 0; i < rcount; ++i)
          {
            key_size_t key = rbuff[i];
            int nid = _segmentDescriptor.getNeuronIndex(key);
            int bid = _segmentDescriptor.getBranchIndex(key);
            int sid = _segmentDescriptor.getSegmentIndex(key);
            std::vector<int>& branchOffsets = branchSegmentOffsets[nid];
            if (pass == 0)
            {  // begin set up branchSegmentOffsets
              if ((unsigned)bid + 1 > branchOffsets.size())
              {
                branchOffsets.resize(bid + 1);
                branchOffsets[bid] = sid + 1;
              }
              else if (sid + 1 > branchOffsets[bid])
              {
                branchOffsets[bid] = sid + 1;
              }
            }
            else
            {  // pass==1, set up fseek, fwrite patterns by rank
              int offset = neuronSegmentOffsets[nid] + branchOffsets[bid] + sid;
              if (i > 0 &&
                  offset ==
                      buffPattern[lastPattern].first +
                          buffPattern[lastPattern].second)
              {
                ++buffPattern[lastPattern].second;
              }
              else
              {
                buffPattern.push_back(std::pair<int, int>(offset, 1));
                ++lastPattern;
              }
            }
          }
          ++buffsRecvdCount;

          if (buffsRecvdCount + requests.size() <= (unsigned)_nKreceives)
          {
            if (buffsRecvdCount > _nSends)
              riter->first = MPI::COMM_WORLD.Irecv(
                  riter->second, BUFF_SIZE, MPI::DOUBLE, MPI_ANY_SOURCE, 0);
            else if (buffsRecvdCount == _nSends)
              requests[0].first =
                  MPI::COMM_WORLD.Irecv(requests[0].second, BUFF_SIZE,
                                        MPI::DOUBLE, MPI_ANY_SOURCE, 0);
          }
          else
            requests.erase(riter);
          if (pass == 1)
          {
            for (unsigned int i = 0; i < buffPattern.size(); ++i)
              buffPattern[i].first *= sizeof(float);
            _demarshalPatterns[sender].push_back(buffPattern);
          }
        }
        MPI::COMM_WORLD.Barrier();
      }
      _outFile = fopen(outFileName.c_str(), "wb");
      if (_outFile == 0)
      {
        std::cerr << "Problem opening " << outFileName << "!" << std::endl;
        exit(0);
      }
      dataSize = keysRecvdCount;
      if (N_IO_NODES > 1)
      {
        dataSize = 0;
        MPI::Group world_group = MPI::COMM_WORLD.Get_group();
        MPI::Group new_group =
            world_group.Incl(_ioNodes.size(), _ioNodes.data());
        MPI_Comm new_communicator = MPI::COMM_WORLD.Create(new_group);
        MPI::COMM_WORLD.Reduce(&dataSize, &keysRecvdCount, 1,
                               MPI::UNSIGNED_LONG, MPI::SUM, _ioNodes[0]);
      }
      if (_rank == _ioNodes[0])
      {
        if (offline)
        {
          fwrite(&dataSize, sizeof(long), 1, _outFile);
          fwrite(&collectionCount, sizeof(long), 1, _outFile);
        }
        else
        {
          float d = FLT_MAX;
          for (int i = 0; i < dataSize; ++i)
            fwrite(&d, sizeof(float), 1, _outFile);
        }
      }
    }
    for (int i = 0; i < _nBuffs; ++i) delete[] keyBuffs[i];
    delete[] keyBuffs;
    calciumMap.clear();
    _calciumBuffs = new float* [_nBuffs];
    for (int i = 0; i < _nBuffs; ++i) _calciumBuffs[i] = new float[BUFF_SIZE];
    assert(_nBuffs > 0);
  }
}

void CalciumVisualization::finalize(RNG& rng)
{
  if (_isIoNode)
  {
    if (offline)
    {
      rewind(_outFile);
      fwrite(&dataSize, sizeof(long), 1, _outFile);
      fwrite(&collectionCount, sizeof(long), 1, _outFile);
    }
    fclose(_outFile);
  }
}

void CalciumVisualization::dataCollection(Trigger* trigger,
                                          NDPairList* ndPairList)
{
  if (_nBuffs > 0)
  {
    int destIO = _ioNodes[_rank % N_IO_NODES];
    if (!_isIoNode)
    {
      std::vector<std::pair<MPI::Request, float*> > requests(_nBuffs);
      for (int i = 0; i < _nBuffs; ++i) requests[i].second = _calciumBuffs[i];
      std::vector<std::pair<MPI::Request, float*> >::iterator riter;
      MPI::Status status;
      std::vector<std::pair<dyn_var_t*, int> >::iterator
          marshallPatternIter = _marshallPatterns.begin(),
          marshallPatternEnd = _marshallPatterns.end();
      for (riter = requests.begin(); riter != requests.end(); ++riter)
      {
        int scount = 0;
        float* sbuff = riter->second;
        for (; scount < BUFF_SIZE && marshallPatternIter != marshallPatternEnd;
             ++marshallPatternIter)
        {
          int n = marshallPatternIter->second;
          // memcpy(&sbuff[scount], marshallPatternIter->first,
          // n*sizeof(float));
          std::copy(marshallPatternIter->first, marshallPatternIter->first + n,
                    &sbuff[scount]);
          scount += n;
        }
        riter->first =
            MPI::COMM_WORLD.Isend(sbuff, scount, MPI::FLOAT, destIO, 0);
      }
      while (marshallPatternIter != marshallPatternEnd)
      {
        riter = requests.begin();
        while (0 == riter->first.Test(status))
          if (++riter == requests.end()) riter = requests.begin();
        int scount = 0;
        float* sbuff = riter->second;
        for (; scount < BUFF_SIZE && marshallPatternIter != marshallPatternEnd;
             ++marshallPatternIter)
        {
          int n = marshallPatternIter->second;
          // memcpy(&sbuff[scount], marshallPatternIter->first,
          // n*sizeof(float));
          std::copy(marshallPatternIter->first, marshallPatternIter->first + n,
                    &sbuff[scount]);
          scount += n;
        }
        riter->first =
            MPI::COMM_WORLD.Isend(sbuff, scount, MPI::FLOAT, destIO, 0);
      }
    }
    else
    {  // _isIoNode
      if (offline)
      {
        long offset =
            sizeof(long) * 2 + sizeof(float) * collectionCount * dataSize;
        fseek(_outFile, offset, SEEK_SET);
        float d = FLT_MAX;
        for (int i = 0; i < dataSize; ++i)
          fwrite(&d, sizeof(float), 1, _outFile);
      }

      std::vector<int> recvCounts(_size, 0);
      std::vector<std::pair<MPI::Request, float*> > requests(_nBuffs);
      for (int i = 0; i < _nBuffs; ++i) requests[i].second = _calciumBuffs[i];
      std::vector<std::pair<MPI::Request, float*> >::iterator riter;
      MPI::Status status;

      for (riter = requests.begin() + 1; riter != requests.end();
           ++riter)  // leave first for self receive
        riter->first = MPI::COMM_WORLD.Irecv(riter->second, BUFF_SIZE,
                                             MPI::FLOAT, MPI_ANY_SOURCE, 0);

      int buffsRecvdCount = 0;
      while (buffsRecvdCount < _nVreceives)
      {
        int rcount = 0;
        int sender = _rank;
        float* rbuff = _calciumBuffs[0];
        if (buffsRecvdCount < _nSends)
        {  // receive from self
          std::vector<std::pair<dyn_var_t*, int> >::iterator
              marshallPatternIter = _marshallPatterns.begin(),
              marshallPatternEnd = _marshallPatterns.end();
          for (;
               rcount < BUFF_SIZE && marshallPatternIter != marshallPatternEnd;
               ++marshallPatternIter)
          {
            int n = marshallPatternIter->second;
            // memcpy(&rbuff[rcount], marshallPatternIter->first,
            // n*sizeof(float));
            std::copy(marshallPatternIter->first,
                      marshallPatternIter->first + n, &rbuff[rcount]);
            rcount += n;
          }
#ifdef SWAP_BYTE_ORDER
          swapByteOrder(rbuff);
#endif
        }
        else
        {
          riter = requests.begin();
          while (0 == riter->first.Test(status))
            if (++riter == requests.end()) riter = requests.begin();
          sender = status.Get_source();
          rbuff = riter->second;
#ifdef SWAP_BYTE_ORDER
          swapByteOrder(rbuff);
#endif
        }

        std::vector<std::pair<int, int> >& buffPattern =
            _demarshalPatterns[sender][recvCounts[sender]];
        ++recvCounts[sender];
        std::vector<std::pair<int, int> >::iterator patternIter =
                                                        buffPattern.begin(),
                                                    patternEnd =
                                                        buffPattern.end();
        for (; patternIter != patternEnd; ++patternIter)
        {
          long offset = patternIter->first +
                        (offline ? (sizeof(long) * 2 +
                                    sizeof(float) * collectionCount * dataSize)
                                 : 0);
          int number = patternIter->second;
          fseek(_outFile, offset, SEEK_SET);
          fwrite(rbuff, sizeof(float), number, _outFile);
          rbuff += number;
        }
        ++buffsRecvdCount;
        if (buffsRecvdCount + requests.size() <= (unsigned)_nVreceives)
        {
          if (buffsRecvdCount > _nSends)
            riter->first = MPI::COMM_WORLD.Irecv(riter->second, BUFF_SIZE,
                                                 MPI::FLOAT, MPI_ANY_SOURCE, 0);
          else if (buffsRecvdCount == _nSends)
            requests[0].first = MPI::COMM_WORLD.Irecv(
                requests[0].second, BUFF_SIZE, MPI::FLOAT, MPI_ANY_SOURCE, 0);
        }
        else
          requests.erase(riter);
      }
    }
  }
  ++collectionCount;
}

void CalciumVisualization::setUpPointers(
    const String& CG_direction, const String& CG_component,
    NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable,
    Constant* CG_constant, CG_CalciumVisualizationInAttrPSet* CG_inAttrPset,
    CG_CalciumVisualizationOutAttrPSet* CG_outAttrPset)
{
  Ca.push_back(Ca_connect);
}

CalciumVisualization::~CalciumVisualization()
{
  for (int i = 0; i < _nBuffs; ++i)
  {
    delete[] _calciumBuffs[i];
  }
  delete[] _calciumBuffs;
}

void CalciumVisualization::duplicate(
    std::auto_ptr<CalciumVisualization>& dup) const
{
  dup.reset(new CalciumVisualization(*this));
}

void CalciumVisualization::duplicate(std::auto_ptr<Variable>& dup) const
{
  dup.reset(new CalciumVisualization(*this));
}

void CalciumVisualization::duplicate(
    std::auto_ptr<CG_CalciumVisualization>& dup) const
{
  dup.reset(new CalciumVisualization(*this));
}

float CalciumVisualization::swapByteOrder(float* buff)
{
  unsigned base = sizeof(float);
  for (unsigned long ii = 0; ii < BUFF_SIZE; ++ii)
  {
    unsigned char sw[base];
    unsigned char* offset = reinterpret_cast<unsigned char*>(&buff[ii]);
    // std::memcpy(sw, offset, base);
    std::copy(offset, offset + base, sw);
    for (unsigned jj = 0; jj < base; ++jj)
    {
      offset[jj] = sw[base - jj - 1];
    }
  }
}
