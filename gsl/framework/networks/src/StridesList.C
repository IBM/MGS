// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
