#ifndef UniformDiscreteDist_H
#define UniformDiscreteDist_H

#include "Lens.h"
#include "CG_UniformDiscreteDistBase.h"
#include "LensContext.h"
#include <memory>

class UniformDiscreteDist : public CG_UniformDiscreteDistBase
{
   public:
      void userInitialize(LensContext* CG_c, double& n1, double& n2);
      int userExecute(LensContext* CG_c);
      UniformDiscreteDist();
      virtual ~UniformDiscreteDist();
      virtual void duplicate(std::auto_ptr<UniformDiscreteDist>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_UniformDiscreteDistBase>& dup) const;
};

#endif
