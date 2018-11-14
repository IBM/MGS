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

#ifndef InitPhase_H
#define InitPhase_H
#include "PhaseElement.h"
#include "Copyright.h"

#include "Phase.h"
#include <memory>
#include <string>

class InitPhase : public Phase {

   public:
      InitPhase(const std::string& name = "", PhaseElement::machineType mType = PhaseElement::CPU);
      virtual void duplicate(std::unique_ptr<Phase>& rv) const;
      virtual ~InitPhase();
     
      virtual std::string getType() const;
      virtual void addToSimulation(Simulation* sim) const;
};


#endif // InitPhase_H
