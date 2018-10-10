#ifndef CombineNVPairs_H
#define CombineNVPairs_H

#include "Lens.h"
#include "CG_CombineNVPairsBase.h"
#include "LensContext.h"
#include "NDPairList.h"
#include <memory>

class CombineNVPairs : public CG_CombineNVPairsBase
{
   public:
      void userInitialize(LensContext* CG_c, NDPairList*& l, DuplicatePointerArray< Functor >& fl);
      std::unique_ptr<NDPairList> userExecute(LensContext* CG_c);
      CombineNVPairs();
      virtual ~CombineNVPairs();
      virtual void duplicate(std::unique_ptr<CombineNVPairs>& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_CombineNVPairsBase>& dup) const;
};

#endif
