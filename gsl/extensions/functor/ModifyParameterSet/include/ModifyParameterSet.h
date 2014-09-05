#ifndef ModifyParameterSet_H
#define ModifyParameterSet_H

#include "Lens.h"
#include "CG_ModifyParameterSetBase.h"
#include "LensContext.h"
#include "ParameterSet.h"
#include <memory>

class ModifyParameterSet : public CG_ModifyParameterSetBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f1, Functor*& f2);
      std::auto_ptr<ParameterSet> userExecute(LensContext* CG_c);
      ModifyParameterSet();
      virtual ~ModifyParameterSet();
      virtual void duplicate(std::auto_ptr<ModifyParameterSet>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ModifyParameterSetBase>& dup) const;
};

#endif
