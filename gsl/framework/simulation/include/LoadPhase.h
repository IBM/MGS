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

#ifndef LoadPhase_H
#define LoadPhase_H
#include "PhaseElement.h"
#include "Copyright.h"

#include "Phase.h"
#include <memory>
#include <string>

class LoadPhase : public Phase {

   public:
      LoadPhase(const std::string& name = "", PhaseElement::machineType mType = CPU);
      virtual void duplicate(std::auto_ptr<Phase>& rv) const;
      virtual ~LoadPhase();
     
      virtual std::string getType() const;
      virtual void addToSimulation(Simulation* sim) const;
};


#endif // LoadPhase_H
