#ifndef Exp_H
#define Exp_H

#include "Lens.h"
#include "CG_ExpBase.h"
#include "LensContext.h"
#include <memory>

class Exp : public CG_ExpBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Exp();
      virtual ~Exp();
      virtual void duplicate(std::auto_ptr<Exp>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_ExpBase>& dup) const;
};

#endif
