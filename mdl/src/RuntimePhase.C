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

#include "RuntimePhase.h"
#include "Phase.h"
#include "PhaseType.h"
#include <memory>
#include <string>

RuntimePhase::RuntimePhase(const std::string& name, 
			   std::auto_ptr<PhaseType>& phaseType,
			   const std::vector<std::string>& pvn)
   : Phase(name, phaseType, pvn)
{
}

void RuntimePhase::duplicate(std::auto_ptr<Phase>& rv) const
{
   rv.reset(new RuntimePhase(*this));
}

RuntimePhase::~RuntimePhase()
{
}

std::string RuntimePhase::getInternalType() const
{
   return "Runtime";
}
