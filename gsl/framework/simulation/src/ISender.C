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

#include "ISender.h"
#include "MemPattern.h"

ISender::ISender()
{
}

ISender::ISender(Simulation* s)
{
   _simulationPtr = s;
}

MemPattern* ISender::getMemPatterns(std::string phaseName, MemPattern* mpptr)
{
  std::list<DistributableCompCategoryBase*>::iterator it, end=_simulationPtr->_distCatList.end();
  int nbytes, total=0, d=0;
  for (it=_simulationPtr->_distCatList.begin(); it!=end; ++it) {
    nbytes=(*it)->setMemPattern(phaseName, _destination, mpptr);
    if (nbytes) ++mpptr;
    total+=nbytes;
  }
  //std::cerr<<_simulationPtr->getRank()<<" sends "<<total<<" to "<<_destination<<std::endl;
  return mpptr;
}

int ISender::getByteCount(std::string phaseName)
{
  int rval=0;
  std::list<DistributableCompCategoryBase*>::iterator it, end=_simulationPtr->_distCatList.end();
  for (it=_simulationPtr->_distCatList.begin(); it!=end; ++it) {
    MPI_Datatype type;
    MPI_Aint sdispl;
    int nBytes = (*it)->getIndexedBlock(phaseName, _destination, &type, sdispl);
    if (nBytes>0) {
      MPI_Type_free(&type);
      rval+=nBytes;
    }
  }
  //std::cerr<<_simulationPtr->getRank()<<" sends "<<rval<<" to "<<_destination<<std::endl;
  return rval;
}

int ISender::getPatternCount(std::string phaseName)
{
  int rval=0;
  std::list<DistributableCompCategoryBase*>::iterator it, end=_simulationPtr->_distCatList.end();
  for (it=_simulationPtr->_distCatList.begin(); it!=end; ++it) {
    MPI_Datatype type;
    MPI_Aint sdispl;
    int nBytes = (*it)->getIndexedBlock(phaseName, _destination, &type, sdispl);
    if (nBytes>0) {
      MPI_Type_free(&type);
      ++rval;
    }
  }
  return rval;
}

bool ISender::getWSendType(std::string phaseName, MPI_Datatype* type)
{
  bool rval=false;
  MPI_Datatype* sndTypes = new MPI_Datatype[_simulationPtr->_distCatList.size()];
  std::vector<MPI_Aint> sdispls;
  std::list<DistributableCompCategoryBase*>::iterator it, end=_simulationPtr->_distCatList.end();
  int nblocks=0;
  for (it=_simulationPtr->_distCatList.begin(); it!=end; ++it) {
    MPI_Datatype* stype = &sndTypes[nblocks];
    MPI_Aint sdispl;
    if ((*it)->getIndexedBlock(phaseName, _destination, stype, sdispl)) {
      sdispls.push_back(sdispl);
      ++nblocks;
    }
  }
  if (nblocks!=0) {
    rval=true;
    int* sndLengths = new int[nblocks];  
    MPI_Aint* sndDispls = new MPI_Aint[nblocks];
    for (int i=0; i<nblocks; ++i) {
      sndLengths[i]=1;
      sndDispls[i]=sdispls[i];
    }
    
    MPI_Type_free(type);
    MPI_Type_create_struct(nblocks, sndLengths, sndDispls, sndTypes, type);
    MPI_Type_commit(type);
    
    delete [] sndLengths;
    delete [] sndDispls;
  }
  for (int i=0; i<nblocks; ++i) MPI_Type_free(&sndTypes[i]);
  delete [] sndTypes;
  return rval;
}

void ISender::setRank(int memorySpaceId)
{
   _destination = memorySpaceId;
}

void ISender::setSimulationPtr(Simulation* s)
{
   _simulationPtr = s;
}

Simulation* ISender::getSimulationPtr()
{
   return _simulationPtr;
}

ISender::~ISender()
{
}

#endif
