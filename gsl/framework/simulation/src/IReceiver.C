// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005, 2006  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI

#include "IReceiver.h"
#include "MemPattern.h"

#include <list>

#include <iostream>


IReceiver::IReceiver()
{
}

IReceiver::IReceiver(Simulation* s)
{
   _simulationPtr = s;
}

MemPattern* IReceiver::getMemPatterns(std::string phaseName, MemPattern* mpptr)
{
  std::list<IndexedBlockCreator*>::iterator it, end=_indexedBlockCreators.end();
  int nbytes, total=0, d=0;
  for (it=_indexedBlockCreators.begin(); it!=end; ++it) {
    nbytes=(*it)->setMemPattern(phaseName, _source, mpptr);
    if (nbytes) ++mpptr;
    total+=nbytes;
  }
  //std::cerr<<_simulationPtr->getRank()<<" receives "<<total<<" from "<<_source<<std::endl;
  return mpptr;
}

int IReceiver::getByteCount(std::string phaseName)
{
  int rval=0;
  std::list<IndexedBlockCreator*>::iterator it, end=_indexedBlockCreators.end();
  for (it=_indexedBlockCreators.begin(); it!=end; ++it) {
    MPI_Datatype type;
    MPI_Aint rdispl;
    int nBytes = (*it)->getIndexedBlock(phaseName, _source, &type, rdispl);
    if (nBytes>0) {
      MPI_Type_free(&type);
      rval+=nBytes;
    }
  }
  //std::cerr<<_simulationPtr->getRank()<<" receives "<<rval<<" from "<<_source<<std::endl;
  return rval;
}

int IReceiver::getPatternCount(std::string phaseName)
{
  int rval=0;
  std::list<IndexedBlockCreator*>::iterator it, end=_indexedBlockCreators.end();
  for (it=_indexedBlockCreators.begin(); it!=end; ++it) {
    MPI_Datatype type;
    MPI_Aint rdispl;
    int nBytes = (*it)->getIndexedBlock(phaseName, _source, &type, rdispl);
    if (nBytes>0) {
      MPI_Type_free(&type);
      ++rval;
    }
  }
  return rval;
}

bool IReceiver::getWReceiveType(std::string phaseName, MPI_Datatype* type)
{
  bool rval=false;
  MPI_Datatype* rcvTypes = new MPI_Datatype[_indexedBlockCreators.size()];
  std::vector<MPI_Aint> rdispls;
  std::list<IndexedBlockCreator*>::iterator it, end=_indexedBlockCreators.end();
  int nblocks=0;
  for (it=_indexedBlockCreators.begin(); it!=end; ++it) {
    MPI_Datatype* rtype = &rcvTypes[nblocks];
    MPI_Aint rdispl;
    if ((*it)->getIndexedBlock(phaseName, _source, rtype, rdispl)) {
      rdispls.push_back(rdispl);
      ++nblocks;
    }
  }
  if (nblocks!=0) {
    rval=true;
    int* rcvLengths = new int[nblocks];  
    MPI_Aint* rcvDispls = new MPI_Aint[nblocks];
    for (int i=0; i<nblocks; ++i) {
      rcvLengths[i]=1;
      rcvDispls[i]=rdispls[i];
    }
    
    MPI_Type_free(type);
    MPI_Type_create_struct(nblocks, rcvLengths, rcvDispls, rcvTypes, type); 
    MPI_Type_commit(type);
    
    delete [] rcvLengths;
    delete [] rcvDispls;
  }
  for (int i=0; i<nblocks; ++i) MPI_Type_free(&rcvTypes[i]);
  delete [] rcvTypes;
  return rval;
}

void IReceiver::setRank(int memorySpaceId)
{
   _source = memorySpaceId;
}

void IReceiver::setSimulationPtr(Simulation* s)
{
   _simulationPtr = s;
   _currentDemarshaller = _demarshallers.begin();
}

void IReceiver::initialize(Simulation* s, int rank)
{
   Demarshaller* dm;
   IndexedBlockCreator* ibc;
   _simulationPtr = s;
   _source = rank;

   // build up CompCategory Demarshaller List
   std::list<DistributableCompCategoryBase*>::iterator iter, end =  _simulationPtr->_distCatList.end();

   for (iter = _simulationPtr->_distCatList.begin(); iter != end; ++iter) {
      if (dm = (*iter)->getDemarshaller(_source)) _demarshallers.push_back(dm);
      if (ibc = (*iter)->getReceiveBlockCreator(_source)) _indexedBlockCreators.push_back(ibc);
   }
   _currentDemarshaller = _demarshallers.begin();
}

IReceiver::~IReceiver()
{
}

#endif
