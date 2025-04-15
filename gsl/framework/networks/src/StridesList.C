// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "StridesList.h"

StridesList::StridesList(StridesList* sl)
{
   _steps = sl->_steps;
   _strides = sl->_strides;
   _order = sl->_order;
};

const std::list<int> & StridesList::getSteps()
{
   return _steps;
};

const std::list<int> & StridesList::getStrides()
{
   return _strides;
};

const std::list<int> & StridesList::getOrder()
{
   return _order;
}
