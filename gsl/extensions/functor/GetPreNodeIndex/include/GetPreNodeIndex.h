#ifndef GetPreNodeIndex_H
#define GetPreNodeIndex_H

#include "Lens.h"
#include "CG_GetPreNodeIndexBase.h"
#include "LensContext.h"
#include <memory>

class GetPreNodeIndex : public CG_GetPreNodeIndexBase
{
   public:
      void userInitialize(LensContext* CG_c);
      int userExecute(LensContext* CG_c);
      GetPreNodeIndex();
      virtual ~GetPreNodeIndex();
      virtual void duplicate(std::unique_ptr<GetPreNodeIndex>& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GetPreNodeIndexBase>& dup) const;
};

#endif
