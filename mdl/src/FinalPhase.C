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

#include "FinalPhase.h"
#include "Phase.h"
#include "PhaseType.h"
#include <memory>
#include <string>

FinalPhase::FinalPhase(const std::string& name, 
		       std::auto_ptr<PhaseType>& phaseType,
		       const std::vector<std::string>& pvn)
   : Phase(name, phaseType, pvn)
{
}

void FinalPhase::duplicate(std::auto_ptr<Phase>& rv) const
{
   rv.reset(new FinalPhase(*this));
}

FinalPhase::~FinalPhase()
{
}

std::string FinalPhase::getInternalType() const
{
   return "Final";
}
