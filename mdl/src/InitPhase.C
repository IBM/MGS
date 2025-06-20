// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "InitPhase.h"
#include "Phase.h"
#include "PhaseType.h"
#include <memory>
#include <string>

InitPhase::InitPhase(const std::string& name, 
		     std::unique_ptr<PhaseType>&& phaseType,
		     const std::vector<std::string>& pvn)
   : Phase(name, std::move(phaseType), pvn)
{
}

void InitPhase::duplicate(std::unique_ptr<Phase>&& rv) const
{
   rv.reset(new InitPhase(*this));
}

InitPhase::~InitPhase()
{
}

std::string InitPhase::getInternalType() const
{
   return "Init";
}
