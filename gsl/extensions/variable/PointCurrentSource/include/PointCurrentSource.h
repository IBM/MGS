// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef PointCurrentSource_H
#define PointCurrentSource_H

#include "Lens.h"
#include "CG_PointCurrentSource.h"
#include <memory>

class PointCurrentSource : public CG_PointCurrentSource
{
   public:
      virtual void stimulate(RNG& rng);
      virtual void setCurrent(Trigger* trigger, NDPairList* ndPairList);
      PointCurrentSource();
      virtual ~PointCurrentSource();
      virtual void duplicate(std::unique_ptr<PointCurrentSource>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_PointCurrentSource>&& dup) const;
};

#endif
