// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NODETYPESET_H
#define NODETYPESET_H
#include "Copyright.h"

#include <list>


class GridLayerDescriptor;

class NodeTypeSet
{
   public:
      NodeTypeSet() {}
      NodeTypeSet(const NodeTypeSet& nts);

   protected:
      std::list<GridLayerDescriptor*> _layers;
};
#endif
