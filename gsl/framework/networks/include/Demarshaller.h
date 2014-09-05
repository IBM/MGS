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
   virtual int demarshall(const char * buffer, int size) = 0; // returns bytes remaining
   virtual ~Demarshaller() {}
};
#endif
#endif

