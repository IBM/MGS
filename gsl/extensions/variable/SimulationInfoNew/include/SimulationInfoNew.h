// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SimulationInfoNew_H
#define SimulationInfoNew_H

#include "Mgs.h"
#include "CG_SimulationInfoNew.h"
#include <memory>

class SimulationInfoNew : public CG_SimulationInfoNew
{
   public:
      void initialize(RNG& rng);
      void calculateInfo(RNG& rng);
      SimulationInfoNew();
      virtual ~SimulationInfoNew();
      virtual void duplicate(std::unique_ptr<SimulationInfoNew>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SimulationInfoNew>&& dup) const;
};

#endif
