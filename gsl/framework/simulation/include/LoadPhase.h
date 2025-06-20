// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef LoadPhase_H
#define LoadPhase_H
#include "PhaseElement.h"
#include "Copyright.h"

#include "Phase.h"
#include <memory>
#include <string>

class LoadPhase : public Phase {

   public:
      LoadPhase(const std::string& name = "", machineType mType = machineType::CPU);
      virtual void duplicate(std::unique_ptr<Phase>& rv) const;
      virtual ~LoadPhase();
     
      virtual std::string getType() const;
      virtual void addToSimulation(Simulation* sim) const;
};


#endif // LoadPhase_H
