// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef STRIDESLIST_H
#define STRIDESLIST_H
#include "Copyright.h"

#include <list>


class StridesList

{
   public:
      StridesList() {};
      StridesList(StridesList* sl);
      const std::list<int> & getSteps();
      const std::list<int> & getStrides();
      const std::list<int> & getOrder();

   protected:
      std::list<int>  _steps;
      std::list<int>  _strides;
      std::list<int>  _order;
};
#endif
