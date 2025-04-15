// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Lens.h"
#include "UniformDiscreteDist.h"
#include "CG_UniformDiscreteDistBase.h"
#include "LensContext.h"
#include "rndm.h"
#include "Simulation.h"
#include <memory>

void UniformDiscreteDist::userInitialize(LensContext* CG_c, double& n1, double& n2) 
{
}

int UniformDiscreteDist::userExecute(LensContext* CG_c) 
{
  return irandom(int(init.n1), int(init.n2), CG_c->sim->getSharedFunctorRandomSeedGenerator());
}

UniformDiscreteDist::UniformDiscreteDist() 
   : CG_UniformDiscreteDistBase()
{
}

UniformDiscreteDist::~UniformDiscreteDist() 
{
}

void UniformDiscreteDist::duplicate(std::unique_ptr<UniformDiscreteDist>&& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

void UniformDiscreteDist::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

void UniformDiscreteDist::duplicate(std::unique_ptr<CG_UniformDiscreteDistBase>&& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

