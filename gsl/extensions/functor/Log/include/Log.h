#ifndef Log_H
#define Log_H

#include "Lens.h"
#include "CG_LogBase.h"
#include "LensContext.h"
#include <memory>

class Log : public CG_LogBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f);
      double userExecute(LensContext* CG_c);
      Log();
      virtual ~Log();
      virtual void duplicate(std::auto_ptr<Log>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_LogBase>& dup) const;
};

#endif
