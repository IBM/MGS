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

#ifndef NODEACCESSOR_H
#define NODEACCESSOR_H
#include "Copyright.h"

#include <memory>
#include <vector>
#include <string>

class Node;
class NodeDescriptor;
class GridLayerDescriptor;

class NodeAccessor
{

   public:
      virtual NodeDescriptor* getNodeDescriptor(std::vector<int> const & coords, int densityIndex) =0;
      virtual NodeDescriptor* getNodeDescriptor(int nodeIndex, int densityIndex) =0;
      virtual void duplicate(std::auto_ptr<NodeAccessor> & r_aptr) const =0;
      virtual int getNbrUnits() =0;
      virtual GridLayerDescriptor* getGridLayerDescriptor() =0;
      virtual std::string getModelName() =0;
      virtual ~NodeAccessor() {}

      static const char* DENSITY_ERROR_MESSAGE;
      static const char* OFFSET_ERROR_MESSAGE;
};
#endif
