// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef NodePartitionItem_H
#define NodePartitionItem_H
#include "Copyright.h"

class NodePartitionItem
{
   public:
      NodePartitionItem();
      virtual ~NodePartitionItem();
      int startIndex;                  // starting index within grid layer
      int endIndex;                    // ending index within grid layer
};

#endif
