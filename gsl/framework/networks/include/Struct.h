// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef STRUCT_H
#define STRUCT_H
#include "Copyright.h"

#include <memory>
#include <vector>

class DataItem;
class LensContext;
class NDPairList;

class Struct
{

   public:
      Struct();
      virtual void duplicate(std::unique_ptr<Struct>&& dup) const=0;
      void initialize(LensContext *c, const std::vector<DataItem*>& args);
      void initialize(const NDPairList& ndplist);
      virtual ~Struct();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args) = 0;
      virtual void doInitialize(const NDPairList& ndplist) = 0;
};

#endif
