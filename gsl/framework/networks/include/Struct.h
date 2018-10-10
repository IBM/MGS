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
      virtual void duplicate(std::unique_ptr<Struct>& dup) const=0;
      void initialize(LensContext *c, const std::vector<DataItem*>& args);
      void initialize(const NDPairList& ndplist);
      virtual ~Struct();
   protected:
      virtual void doInitialize(LensContext *c, 
				const std::vector<DataItem*>& args) = 0;
      virtual void doInitialize(const NDPairList& ndplist) = 0;
};

#endif
