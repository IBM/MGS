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
#include "Mdl.h"

#include "Phase.h"
#include <memory>
#include <string>

class PhaseType;

class LoadPhase : public Phase {

   public:
      LoadPhase(const std::string& name, std::auto_ptr<PhaseType>& phaseType,
		const std::vector<std::string>& pvn);
      virtual void duplicate(std::auto_ptr<Phase>& rv) const;
      virtual ~LoadPhase();    
   protected:
      virtual std::string getInternalType() const;
};

#endif // LoadPhase_H
