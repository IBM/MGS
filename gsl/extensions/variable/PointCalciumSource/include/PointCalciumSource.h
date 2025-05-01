// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PointCalciumSource_H
#define PointCalciumSource_H

#include "Mgs.h"
#include "CG_PointCalciumSource.h"
#include <memory>

class PointCalciumSource : public CG_PointCalciumSource
{
   public:
      void stimulate(RNG& rng);
      virtual void setCaCurrent(Trigger* trigger, NDPairList* ndPairList);
      PointCalciumSource();
      virtual ~PointCalciumSource();
      virtual void duplicate(std::unique_ptr<PointCalciumSource>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_PointCalciumSource>&& dup) const;
};

#endif
