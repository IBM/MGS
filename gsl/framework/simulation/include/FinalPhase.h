// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef FinalPhase_H
#define FinalPhase_H
#include "Copyright.h"

#include "Phase.h"
#include <memory>
#include <string>

class FinalPhase : public Phase {

   public:
      FinalPhase(const std::string& name = "");
      virtual void duplicate(std::auto_ptr<Phase>& rv) const;
      virtual ~FinalPhase();
     
      virtual std::string getType() const;
      virtual void addToSimulation(Simulation* sim) const;
};


#endif // FinalPhase_H
