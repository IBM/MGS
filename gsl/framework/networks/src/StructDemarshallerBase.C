// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
 
int StructDemarshallerBase::demarshall(const char * buffer, int size, bool& rebuildRequested) 
{
  const char* buff = buffer;
  int buffSize = size;
  std::vector<Demarshaller*>::iterator end = _demarshallers.end();
  while( _demarshallersIter != end && buffSize !=0)
  {
      buffSize = (*_demarshallersIter)->demarshall(buff, buffSize, rebuildRequested);
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
