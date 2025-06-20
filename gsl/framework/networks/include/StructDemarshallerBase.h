// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef STRUCTDEMARSHALLERBASE_H
#define STRUCTDEMARSHALLERBASE_H
#include "Copyright.h"
#ifdef HAVE_MPI
#include <mpi.h>
#include "Demarshaller.h"
#include <vector>

class StructDemarshallerBase : public Demarshaller 

{
public:
   StructDemarshallerBase();
   virtual void reset();
   virtual bool done(); 
   virtual void getBlocks(std::vector<int> & blengths, std::vector<MPI_Aint> & blocs);
   virtual int demarshall(const char * buffer, int size, bool& rebuildRequested);
   ~StructDemarshallerBase();

protected:
   std::vector<Demarshaller*> _demarshallers;
   std::vector<Demarshaller*>::iterator _demarshallersIter;
};

#endif
#endif
