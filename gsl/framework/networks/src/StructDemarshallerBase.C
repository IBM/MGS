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

#include "StructDemarshallerBase.h"

#ifdef HAVE_MPI
#include <iostream>

StructDemarshallerBase::StructDemarshallerBase()
{
}

void StructDemarshallerBase::reset()
{
  std::vector<Demarshaller*>::iterator iter = _demarshallersIter = _demarshallers.begin();
  std::vector<Demarshaller*>::iterator end = _demarshallers.end();
  for (; iter != end; ++iter) {
    (*iter)->reset();
  }
}

bool StructDemarshallerBase::done() 
{
  return (_demarshallersIter == _demarshallers.end());
}
 
int StructDemarshallerBase::demarshall(const char * buffer, int size) 
{
  const char* buff = buffer;
  int buffSize = size;
  std::vector<Demarshaller*>::iterator end = _demarshallers.end();
  while( _demarshallersIter != end && buffSize !=0)
  {
      buffSize = (*_demarshallersIter)->demarshall(buff, buffSize);
      buff = buffer+(size-buffSize);
      if ((*_demarshallersIter)->done()){
         ++_demarshallersIter;
      }
  }
  return buffSize;
}

void StructDemarshallerBase::getBlocks(std::vector<int> & blengths, std::vector<MPI_Aint> & blocs)
{
  std::vector<Demarshaller*>::iterator iter, end = _demarshallers.end();
  for (iter=_demarshallers.begin(); iter!=end; ++iter) {
    (*iter)->getBlocks(blengths, blocs);
  }
}

StructDemarshallerBase::~StructDemarshallerBase()
{
}

#endif
