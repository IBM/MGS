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

void UniformDiscreteDist::duplicate(std::unique_ptr<UniformDiscreteDist>& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

void UniformDiscreteDist::duplicate(std::unique_ptr<Functor>& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

void UniformDiscreteDist::duplicate(std::unique_ptr<CG_UniformDiscreteDistBase>& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

