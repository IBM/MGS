// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef FinalPhase_H
#define FinalPhase_H
#include "Mdl.h"

#include "Phase.h"
#include <memory>
#include <string>

class PhaseType;

class FinalPhase : public Phase {

   public:
      FinalPhase(const std::string& name, std::unique_ptr<PhaseType>&& phaseType,
		 const std::vector<std::string>& pvn);
      virtual void duplicate(std::unique_ptr<Phase>&& rv) const;
      virtual ~FinalPhase();    
   protected:
      virtual std::string getInternalType() const;
};

#endif // FinalPhase_H
