// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "LoadPhase.h"
#include "Phase.h"
#include "PhaseType.h"
#include <memory>
#include <string>

LoadPhase::LoadPhase(const std::string& name, 
		     std::unique_ptr<PhaseType>&& phaseType,
		     const std::vector<std::string>& pvn)
   : Phase(name, std::move(phaseType), pvn)
{
}

void LoadPhase::duplicate(std::unique_ptr<Phase>&& rv) const
{
   rv.reset(new LoadPhase(*this));
}

LoadPhase::~LoadPhase()
{
}

std::string LoadPhase::getInternalType() const
{
   return "Load";
}
