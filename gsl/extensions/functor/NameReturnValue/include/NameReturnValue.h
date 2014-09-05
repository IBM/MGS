#ifndef NameReturnValue_H
#define NameReturnValue_H

#include "Lens.h"
#include "CG_NameReturnValueBase.h"
#include "LensContext.h"
#include "NDPairList.h"
#include <memory>

class NameReturnValue : public CG_NameReturnValueBase
{
   public:
      void userInitialize(LensContext* CG_c, String& s, Functor*& f);
      std::auto_ptr<NDPairList> userExecute(LensContext* CG_c);
      NameReturnValue();
      virtual ~NameReturnValue();
      virtual void duplicate(std::auto_ptr<NameReturnValue>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_NameReturnValueBase>& dup) const;
};

#endif
