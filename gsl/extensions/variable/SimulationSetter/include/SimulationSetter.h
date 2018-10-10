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
