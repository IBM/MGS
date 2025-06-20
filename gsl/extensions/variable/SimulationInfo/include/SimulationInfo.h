// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef SimulationInfo_H
#define SimulationInfo_H

#include "Mgs.h"
#include "CG_SimulationInfo.h"
#include <memory>

class SimulationInfo : public CG_SimulationInfo
{
   public:
      void initialize(RNG& rng);
      void calculateInfo(RNG& rng);
      SimulationInfo();
      virtual ~SimulationInfo();
      virtual void duplicate(std::unique_ptr<SimulationInfo>&& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SimulationInfo>&& dup) const;
};

#endif
