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

#ifndef InitPhase_H
#define InitPhase_H
#include "Mdl.h"

#include "Phase.h"
#include <memory>
#include <string>

class PhaseType;

class InitPhase : public Phase {

   public:
      InitPhase(const std::string& name, std::auto_ptr<PhaseType>& phaseType,
		const std::vector<std::string>& pvn);
      virtual void duplicate(std::auto_ptr<Phase>& rv) const;
      virtual ~InitPhase();    
   protected:
      virtual std::string getInternalType() const;
};


#endif // InitPhase_H
