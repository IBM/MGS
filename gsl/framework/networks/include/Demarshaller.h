// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DEMARSHALLER_H
#define DEMARSHALLER_H
#include "Copyright.h"
#ifdef HAVE_MPI
#include <mpi.h>
#include <vector>

class Demarshaller
{
public:
   virtual void reset() = 0;
   virtual bool done() = 0;
   virtual void getBlocks(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs)
   {
     blengths.resize(1,0);
     blocs.resize(1,0);
   }
   virtual int demarshall(const char * buffer, int size, bool& rebuildRequested) = 0; // returns bytes remaining
   virtual ~Demarshaller() {}
};
#endif
#endif

