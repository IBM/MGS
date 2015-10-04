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
   bool isInSpace(double key);
   bool areInSpace(double key1, double key2);
   TouchSpace* duplicate();
 private:
   SynapseType _type;
   Params _params;
   bool _autapses;
   SegmentDescriptor _segmentDescriptor;
};
#endif

