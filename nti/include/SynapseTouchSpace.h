// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SYNAPSETOUCHSPACE_H
#define SYNAPSETOUCHSPACE_H
#include "TouchSpace.h"
#include "Params.h"
#include "SegmentDescriptor.h"

#include <map>
#include <list>

#include <mpi.h>

class SynapseTouchSpace : public TouchSpace
{
 public:
   enum SynapseType {ELECTRICAL, CHEMICAL};
   SynapseTouchSpace(SynapseType type,
		     Params* params,
		     bool autapses);
   SynapseTouchSpace(SynapseTouchSpace& synapseTouchSpace);
   ~SynapseTouchSpace();
   bool isInSpace(key_size_t key);
   bool areInSpace(key_size_t key1, key_size_t key2);
   TouchSpace* duplicate();
 private:
   SynapseType _type;
   Params _params;
   bool _autapses;
   SegmentDescriptor _segmentDescriptor;
};
#endif

