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
#include "VoltageVisualization.h"
#include "Simulation.h"
#include "CG_VoltageVisualization.h"
#include "NeuronPartitioner.h"

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

VoltageVisualization::VoltageVisualization()
  : CG_VoltageVisualization(), _outFile(0), _rank(-1), _size(-1), _isIoNode(false),
    _voltageBuffs(0), _nBuffs(0), _nSends(0), _nKreceives(0), _nVreceives(0)
{
} 

void VoltageVisualization::initialize(RNG& rng) 
{
  _rank=getSimulation().getRank();
  _size=getSimulation().getNumProcesses();
  assert(_size>=N_IO_NODES);
  _ioNodes.resize(N_IO_NODES);
  for (int i=0; i<N_IO_NODES; ++i) {
    _ioNodes[i]=i; // randomize this for better performance
  }
  std::vector<int>::iterator iter=find(_ioNodes.begin(), _ioNodes.end(), _rank);
  _isIoNode=(iter!=_ioNodes.end());

  // Note: The use of new here is not thread safe! 
  // Therefore, this variable object should be instantiated only once
  // per memory space, and initialized in its own InitPhase

  assert(deltaT);
  std::map<CompartmentKey, float*, CompartmentKey::compare> voltageMap;
  if (V.size()>0) {
    for (int i=0; i<V.size(); ++i) {
      double key=branchData[i]->key;
      int sid=_segmentDescriptor.getSegmentIndex(key);
      int Vsz=V[i]->size();
      for (int j=0; j<Vsz; ++j) {
        voltageMap[CompartmentKey(key,j+sid)]=&(*V[i])[j];
      }
    }
    V.clear();
    branchData.clear();
  }

  int localDataSize=voltageMap.size();
  unsigned long keysRecvdCount=0;

  _nSends = int(ceil(double(localDataSize)/double(BUFF_SIZE)));
  int destIO=_ioNodes[_rank % N_IO_NODES];
  for (int i=0; i<N_IO_NODES; ++i) {
    int sendToDest=0;
    if (i==_rank % N_IO_NODES) sendToDest=_nSends;
    MPI::COMM_WORLD.Reduce(&sendToDest, &_nVreceives, 1, MPI::INT, MPI::SUM, _ioNodes[i]);
    MPI::COMM_WORLD.Reduce(&_nSends, &_nKreceives, 1, MPI::INT, MPI::SUM, _ioNodes[i]);
  }
  _nBuffs = _isIoNode ? ( (N_RECEIVE_BUFFS<_nVreceives) ? N_RECEIVE_BUFFS : _nVreceives ) : 
                         ( (N_SEND_BUFFS<_nSends) ? N_SEND_BUFFS : _nSends );
  if (_isIoNode) assert(_nBuffs>0);
  double** keyBuffs=0;
  if (_nBuffs>0) {
    keyBuffs=new double*[_nBuffs];
    for (int i=0; i<_nBuffs; ++i) keyBuffs[i]=new double[BUFF_SIZE];
    std::map<CompartmentKey, float*, CompartmentKey::compare>::iterator
      voltageMapIter=voltageMap.begin();
    for (int i=0; i<localDataSize; ++i, ++voltageMapIter) {
      if (i%BUFF_SIZE!=0 && voltageMapIter->second==_marshallPatterns.back().first+1)
        _marshallPatterns.back().second++;
      else
        _marshallPatterns.push_back(std::pair<float*, int>(voltageMapIter->second, 1));
    }
    if (!_isIoNode) {
      for (int pass=0; pass<2; ++pass) {
        std::vector<std::pair<MPI::Request, double*> > requests(_nBuffs);
        std::map<double*, int> destIndices;
        for (int i=0; i<_nBuffs; ++i) {
          requests[i].second=keyBuffs[i];
          destIndices[keyBuffs[i]]=0;
        }
        std::vector<std::pair<MPI::Request, double*> >::iterator riter, rend;
        MPI::Status status;
        voltageMapIter=voltageMap.begin();
        std::map<CompartmentKey, float*, CompartmentKey::compare>::iterator 
          voltageMapEnd=voltageMap.end();
        rend=requests.end(); 

        for (riter=requests.begin(); riter!=rend; ++riter) {
          int scount;
          double* sbuff = riter->second;
          assert(destIndices[sbuff]==0);
          for (scount=0;
               scount<BUFF_SIZE && voltageMapIter!=voltageMapEnd;
               ++scount, ++voltageMapIter) {
            sbuff[scount]=_segmentDescriptor.modifySegmentKey(SegmentDescriptor::segmentIndex, 
                                                              voltageMapIter->first._cptIdx, 
                                                              voltageMapIter->first._key);
          }
          riter->first=MPI::COMM_WORLD.Isend(sbuff, scount, MPI::DOUBLE, _ioNodes[destIndices[sbuff]], 0);
          if (++destIndices[sbuff]==N_IO_NODES) destIndices[sbuff]=0;
        }
        while (voltageMapIter!=voltageMapEnd) {
          riter=requests.begin();
          while (0==riter->first.Test(status))
            if (++riter==rend) riter=requests.begin();
          int scount;
          double* sbuff = riter->second;
          if (destIndices[sbuff]==0) {
            for (scount=0;
                 scount<BUFF_SIZE && voltageMapIter!=voltageMapEnd;
                 ++scount, ++voltageMapIter)
              sbuff[scount]=_segmentDescriptor.modifySegmentKey(SegmentDescriptor::segmentIndex, 
                                                                voltageMapIter->first._cptIdx, 
                                                                voltageMapIter->first._key);
          }
          riter->first=MPI::COMM_WORLD.Isend(sbuff, scount, MPI::DOUBLE, _ioNodes[destIndices[sbuff]], 0);
          if (++destIndices[sbuff]==N_IO_NODES) destIndices[sbuff]=0;
        }
        MPI::COMM_WORLD.Barrier();
      }
    } else { // _isIoNode
      _demarshalPatterns.resize(_size);
      int totalNeurons=0;
      int totalSegmentsRead=0;
      int neuronsPerLayer[6]={0,0,0,0,0,0};
      std::vector<int> neuronSegmentOffsets;
      NeuronPartitioner::countAllNeurons(inFileName.c_str(), totalNeurons, 
                                         totalSegmentsRead, neuronsPerLayer,
                                         neuronSegmentOffsets);
      for (int i=totalNeurons-1; i>0; --i)
        neuronSegmentOffsets[i]=neuronSegmentOffsets[i-1];
      neuronSegmentOffsets[0]=0;

      for (int i=1; i<totalNeurons; ++i)
        neuronSegmentOffsets[i]=neuronSegmentOffsets[i]+neuronSegmentOffsets[i-1];

      std::vector<std::vector<int> > branchSegmentOffsets;
      branchSegmentOffsets.resize(totalNeurons);

      for (int pass=0; pass<2; ++pass) {
        keysRecvdCount=localDataSize;
        std::vector<std::pair<MPI::Request, double*> > requests(_nBuffs);
        for (int i=0; i<_nBuffs; ++i) requests[i].second=keyBuffs[i];
        std::vector<std::pair<MPI::Request, double*> >::iterator riter;
        MPI::Status status;
      
        for (riter=requests.begin()+1; riter!=requests.end(); ++riter) // leave first for self receive
          riter->first = MPI::COMM_WORLD.Irecv(riter->second, BUFF_SIZE, MPI::DOUBLE, MPI_ANY_SOURCE, 0);

        if (pass==1) {
          for (int i=0; i<totalNeurons; ++i) { // finish set up of branchSegmentOffsets
            std::vector<int>& branchOffsets=branchSegmentOffsets[i];
            int totalBranches=branchOffsets.size();
            for (int j=totalBranches-1; j>0; --j)
              branchOffsets[j]=branchOffsets[j-1];
            if (totalBranches > 0) branchOffsets[0]=0;
            for (int j=1; j<totalBranches; ++j)
              branchOffsets[j]=branchOffsets[j]+branchOffsets[j-1];
          }
        }

        std::map<CompartmentKey, float*, CompartmentKey::compare>::iterator 
          voltageMapIter=voltageMap.begin(),
          voltageMapEnd=voltageMap.end();
        
        int buffsRecvdCount=0;
        while (buffsRecvdCount<_nKreceives) {
          int rcount, sender = _rank;
          double* rbuff = keyBuffs[0];
          if (buffsRecvdCount<_nSends) { // receive from self
            for (rcount=0;
                 rcount<BUFF_SIZE && voltageMapIter!=voltageMapEnd; 
                 ++rcount, ++voltageMapIter) {
              rbuff[rcount]=_segmentDescriptor.modifySegmentKey(SegmentDescriptor::segmentIndex, 
                                                                voltageMapIter->first._cptIdx, 
                                                                voltageMapIter->first._key);
            }
          }
          else {
            riter=requests.begin();
            while (0==riter->first.Test(status))
              if (++riter==requests.end()) riter=requests.begin();
            sender = status.Get_source();
            rcount = status.Get_count(MPI::DOUBLE);
            rbuff = riter->second;
            keysRecvdCount+=rcount;
          }

          std::vector<std::pair<int ,int> > buffPattern;
          int lastPattern=-1;
          for (int i=0; i<rcount; ++i) {  
            double key=rbuff[i];
            int nid=_segmentDescriptor.getNeuronIndex(key);
            int bid=_segmentDescriptor.getBranchIndex(key);
            int sid=_segmentDescriptor.getSegmentIndex(key);
            std::vector<int>& branchOffsets=branchSegmentOffsets[nid];
            if (pass==0) { // begin set up branchSegmentOffsets
              if (bid+1>branchOffsets.size()) {
                branchOffsets.resize(bid+1);
                branchOffsets[bid]=sid+1;
              }
              else if (sid+1>branchOffsets[bid]) {
                branchOffsets[bid]=sid+1;
              }
            }
            else { // pass==1, set up fseek, fwrite patterns by rank
              int offset=neuronSegmentOffsets[nid]+branchOffsets[bid]+sid;
              if (i>0 && offset==buffPattern[lastPattern].first+buffPattern[lastPattern].second) {
                ++buffPattern[lastPattern].second;
              }
              else {
                buffPattern.push_back(std::pair<int, int>(offset,1));
                ++lastPattern;
              }
            }
          }
          ++buffsRecvdCount;

          if (buffsRecvdCount+requests.size()<=_nKreceives) {
            if (buffsRecvdCount>_nSends)
              riter->first = MPI::COMM_WORLD.Irecv(riter->second, BUFF_SIZE, MPI::DOUBLE, MPI_ANY_SOURCE, 0);
            else if (buffsRecvdCount==_nSends)
              requests[0].first = MPI::COMM_WORLD.Irecv(requests[0].second, BUFF_SIZE, MPI::DOUBLE, MPI_ANY_SOURCE, 0);
          } else
            requests.erase(riter);
          if (pass==1) {
            for (int i=0; i<buffPattern.size(); ++i)
              buffPattern[i].first*=sizeof(float);
            _demarshalPatterns[sender].push_back(buffPattern);
          }
        }
        MPI::COMM_WORLD.Barrier();
      }

      _outFile = fopen(outFileName.c_str(), "wb");
      if (_outFile==0) {
        std::cerr<<"Problem opening "<<outFileName<<"!"<<std::endl;
        exit(0);
      }

      dataSize=keysRecvdCount;
      if (N_IO_NODES>1) {
        dataSize=0;
        MPI::Group world_group=MPI::COMM_WORLD.Get_group();
        MPI::Group new_group=world_group.Incl(_ioNodes.size(), _ioNodes.data());
        MPI_Comm new_communicator=MPI::COMM_WORLD.Create(new_group);
        MPI::COMM_WORLD.Reduce(&dataSize, &keysRecvdCount, 1, MPI::UNSIGNED_LONG, MPI::SUM, _ioNodes[0]);
      }
      if (_rank==_ioNodes[0]) {
        if (offline) {
          fwrite(&dataSize, sizeof(long), 1, _outFile);
          fwrite(&collectionCount, sizeof(long), 1, _outFile);
        } else {
          float d=FLT_MAX;
          for (int i=0; i<dataSize; ++i) 
            fwrite(&d, sizeof(float), 1, _outFile);
        }
      }
    }

    for (int i=0; i<_nBuffs; ++i) delete [] keyBuffs[i];
    delete [] keyBuffs;
    voltageMap.clear();
    _voltageBuffs=new float*[_nBuffs];
    for (int i=0; i<_nBuffs; ++i) _voltageBuffs[i]=new float[BUFF_SIZE];
    assert(_nBuffs>0);
  }
}

void VoltageVisualization::finalize(RNG& rng) 
{
  if (_isIoNode) {
    if (offline) {
      rewind(_outFile);
      fwrite(&dataSize, sizeof(long), 1, _outFile);
      fwrite(&collectionCount, sizeof(long), 1, _outFile);
    }
    fclose(_outFile);
  }
}

void VoltageVisualization::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  if (_nBuffs>0) {
    int destIO=_ioNodes[_rank % N_IO_NODES];
    if (!_isIoNode) {
      std::vector<std::pair<MPI::Request, float*> > requests(_nBuffs);
      for (int i=0; i<_nBuffs; ++i) 
        requests[i].second=_voltageBuffs[i];
      std::vector<std::pair<MPI::Request, float*> >::iterator riter;
      MPI::Status status;
      std::vector<std::pair<float*, int> >::iterator
        marshallPatternIter=_marshallPatterns.begin(),
        marshallPatternEnd=_marshallPatterns.end();
      for (riter=requests.begin(); riter!=requests.end(); ++riter) {
        int scount=0;
        float* sbuff = riter->second;
        for (; scount<BUFF_SIZE && 
               marshallPatternIter!=marshallPatternEnd; ++marshallPatternIter) {
          int n=marshallPatternIter->second;
          memcpy(&sbuff[scount], marshallPatternIter->first, n*sizeof(float));
          scount+=n;
        }
        riter->first=MPI::COMM_WORLD.Isend(sbuff, scount, MPI::FLOAT, destIO, 0);
      }
      while (marshallPatternIter!=marshallPatternEnd) {
        riter=requests.begin();
        while (0==riter->first.Test(status))
          if (++riter==requests.end()) riter=requests.begin();
        int scount=0;
        float* sbuff = riter->second;
        for (; scount<BUFF_SIZE && 
               marshallPatternIter!=marshallPatternEnd; ++marshallPatternIter) {
          int n=marshallPatternIter->second;
          memcpy(&sbuff[scount], marshallPatternIter->first, n*sizeof(float));
          scount+=n;
        }
        riter->first=MPI::COMM_WORLD.Isend(sbuff, scount, MPI::FLOAT, destIO, 0);
      }
    }
    else { // _isIoNode
      if (offline) {
        long offset=sizeof(long)*2 + sizeof(float)*collectionCount*dataSize;
        fseek(_outFile, offset, SEEK_SET);
        float d=FLT_MAX;
        for (int i=0; i<dataSize; ++i) 
          fwrite(&d, sizeof(float), 1, _outFile);
      }

      std::vector<int> recvCounts(_size,0);
      std::vector<std::pair<MPI::Request, float*> > requests(_nBuffs);
      for (int i=0; i<_nBuffs; ++i) requests[i].second=_voltageBuffs[i];
      std::vector<std::pair<MPI::Request, float*> >::iterator riter;
      MPI::Status status;

      for (riter=requests.begin()+1; riter!=requests.end(); ++riter) // leave first for self receive
        riter->first = MPI::COMM_WORLD.Irecv(riter->second, BUFF_SIZE, MPI::FLOAT, MPI_ANY_SOURCE, 0);

      int buffsRecvdCount=0;
      while (buffsRecvdCount<_nVreceives) {
        int rcount = 0;
        int sender = _rank;
        float* rbuff = _voltageBuffs[0];
        if (buffsRecvdCount<_nSends) { // receive from self
          std::vector<std::pair<float*, int> >::iterator
            marshallPatternIter=_marshallPatterns.begin(),
            marshallPatternEnd=_marshallPatterns.end();
          for (; rcount<BUFF_SIZE && 
                 marshallPatternIter!=marshallPatternEnd; ++marshallPatternIter) {
            int n=marshallPatternIter->second;
            memcpy(&rbuff[rcount], marshallPatternIter->first, n*sizeof(float));
            rcount+=n;
          }
#ifdef SWAP_BYTE_ORDER
          swapByteOrder(rbuff);
#endif
        }
        else {
          riter=requests.begin();
          while (0==riter->first.Test(status))
            if (++riter==requests.end()) riter=requests.begin();
          sender = status.Get_source();
          rbuff = riter->second;
#ifdef SWAP_BYTE_ORDER
          swapByteOrder(rbuff);
#endif
        }

        std::vector<std::pair<int ,int> >& buffPattern=_demarshalPatterns[sender][recvCounts[sender]];
        ++recvCounts[sender];
        std::vector<std::pair<int ,int> >::iterator 
          patternIter=buffPattern.begin(),
          patternEnd=buffPattern.end();
        for (; patternIter!=patternEnd; ++patternIter) {
          long offset=patternIter->first + (offline ? (sizeof(long)*2 + sizeof(float)*collectionCount*dataSize) : 0);
          int number=patternIter->second;
          fseek(_outFile, offset, SEEK_SET);
          fwrite(rbuff, sizeof(float), number, _outFile);
          rbuff+=number;
        }
        ++buffsRecvdCount;
        if (buffsRecvdCount+requests.size()<=_nVreceives) {
          if (buffsRecvdCount>_nSends)
            riter->first = MPI::COMM_WORLD.Irecv(riter->second, BUFF_SIZE, MPI::FLOAT, MPI_ANY_SOURCE, 0);
          else if (buffsRecvdCount==_nSends)
            requests[0].first = MPI::COMM_WORLD.Irecv(requests[0].second, BUFF_SIZE, MPI::FLOAT, MPI_ANY_SOURCE, 0);
        }
        else requests.erase(riter);
      }
    }
  }
  ++collectionCount;
}

void VoltageVisualization::setUpPointers(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_VoltageVisualizationInAttrPSet* CG_inAttrPset, CG_VoltageVisualizationOutAttrPSet* CG_outAttrPset) 
{
  V.push_back(V_connect);
}

VoltageVisualization::~VoltageVisualization() 
{
  for (int i=0; i<_nBuffs; ++i) {
    delete [] _voltageBuffs[i];
  }
  delete [] _voltageBuffs;
}

void VoltageVisualization::duplicate(std::auto_ptr<VoltageVisualization>& dup) const
{
   dup.reset(new VoltageVisualization(*this));
}

void VoltageVisualization::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new VoltageVisualization(*this));
}

void VoltageVisualization::duplicate(std::auto_ptr<CG_VoltageVisualization>& dup) const
{
   dup.reset(new VoltageVisualization(*this));
}

float VoltageVisualization::swapByteOrder(float* buff)
{
  unsigned base=sizeof(float);
  for (unsigned long ii=0; ii<BUFF_SIZE; ++ii) {
    unsigned char sw[base];
    unsigned char* offset = reinterpret_cast<unsigned char*>(&buff[ii]);
    std::memcpy(sw, offset, base);
    for (unsigned jj=0; jj<base; ++jj) {
      offset[jj]=sw[base-jj-1];
    }
  }
}
