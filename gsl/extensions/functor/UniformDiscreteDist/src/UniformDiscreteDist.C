#include "Lens.h"
#include "UniformDiscreteDist.h"
#include "CG_UniformDiscreteDistBase.h"
#include "LensContext.h"
#include "rndm.h"
#include <memory>

void UniformDiscreteDist::userInitialize(LensContext* CG_c, double& n1, double& n2) 
{
}

int UniformDiscreteDist::userExecute(LensContext* CG_c) 
{
  return irandom(int(init.n1), int(init.n2));
}

UniformDiscreteDist::UniformDiscreteDist() 
   : CG_UniformDiscreteDistBase()
{
}

UniformDiscreteDist::~UniformDiscreteDist() 
{
}

void UniformDiscreteDist::duplicate(std::auto_ptr<UniformDiscreteDist>& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

void UniformDiscreteDist::duplicate(std::auto_ptr<Functor>& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

void UniformDiscreteDist::duplicate(std::auto_ptr<CG_UniformDiscreteDistBase>& dup) const
{
   dup.reset(new UniformDiscreteDist(*this));
}

