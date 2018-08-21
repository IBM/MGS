// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

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

