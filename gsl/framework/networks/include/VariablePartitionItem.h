// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef VariablePartitionItem_H
#define VariablePartitionItem_H
#include "Copyright.h"

class VariablePartitionItem
{
   public:
      VariablePartitionItem();
      virtual ~VariablePartitionItem();
      int startIndex;     // starting index 
      int endIndex;       // ending index 
};

#endif
