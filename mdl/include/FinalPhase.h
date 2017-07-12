// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef FinalPhase_H
#define FinalPhase_H
#include "Mdl.h"

#include "Phase.h"
#include <memory>
#include <string>

class PhaseType;

class FinalPhase : public Phase {

   public:
      FinalPhase(const std::string& name, std::auto_ptr<PhaseType>& phaseType,
		 const std::vector<std::string>& pvn);
      virtual void duplicate(std::auto_ptr<Phase>& rv) const;
      virtual ~FinalPhase();    
   protected:
      virtual std::string getInternalType() const;
};

#endif // FinalPhase_H
