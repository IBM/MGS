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

#ifndef RuntimePhase_H
#define RuntimePhase_H
#include "Mdl.h"

#include "Phase.h"
#include <memory>
#include <string>

class PhaseType;

class RuntimePhase : public Phase {

   public:
      RuntimePhase(const std::string& name, 
		   std::auto_ptr<PhaseType>& phaseType,
		   const std::vector<std::string>& pvn);
      virtual void duplicate(std::auto_ptr<Phase>& rv) const;
      virtual ~RuntimePhase();    
   protected:
      virtual std::string getInternalType() const;
};


#endif // RuntimePhase_H
