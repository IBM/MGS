// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
