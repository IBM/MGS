// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
