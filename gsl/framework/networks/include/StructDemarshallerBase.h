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
   virtual int demarshall(const char * buffer, int size);
   ~StructDemarshallerBase();

protected:
   std::vector<Demarshaller*> _demarshallers;
   std::vector<Demarshaller*>::iterator _demarshallersIter;
};

#endif
#endif
