// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef RuntimePhase_H
#define RuntimePhase_H
#include "PhaseElement.h"
#include "Copyright.h"

#include "Phase.h"
#include <memory>
#include <string>

class RuntimePhase : public Phase {

   public:
  RuntimePhase(const std::string& name = "", machineType mType = machineType::CPU);
      virtual void duplicate(std::unique_ptr<Phase>& rv) const;
      virtual ~RuntimePhase();
     
      virtual std::string getType() const;
      virtual void addToSimulation(Simulation* sim) const;
};


#endif // RuntimePhase_H
