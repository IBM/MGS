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

#ifndef INDEXEDBLOCKCREATOR_H
#define INDEXEDBLOCKCREATOR_H
#include "Copyright.h"
#ifdef HAVE_MPI
#include <mpi.h>
#include <string>

class MemPattern;

class IndexedBlockCreator
{
public:
   virtual int getIndexedBlock(std::string phaseName, int, MPI_Datatype* blockType, MPI_Aint& blockLocation) = 0;
   virtual int setMemPattern(std::string phaseName, int, MemPattern* mp) = 0;
   virtual ~IndexedBlockCreator() {}
};
#endif
#endif
