// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SimulationSetter_H
#define SimulationSetter_H

#include "Lens.h"
#include "CG_SimulationSetter.h"
#include <memory>

class SimulationSetter : public CG_SimulationSetter
{
   public:
      void initialize(RNG& rng);
      void propagateToggles(RNG& rng);
      virtual void switchPlasticityOnOff(Trigger* trigger, NDPairList* ndPairList);
      SimulationSetter();
      virtual ~SimulationSetter();
      virtual void duplicate(std::unique_ptr<SimulationSetter>& dup) const;
      virtual void duplicate(std::unique_ptr<Variable>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_SimulationSetter>& dup) const;
};

#endif
