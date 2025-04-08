#ifndef RandomDispersalLayout_H
#define RandomDispersalLayout_H

#include "Lens.h"
#include "CG_RandomDispersalLayoutBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include "rndm.h"
#include <memory>

class RandomDispersalLayout : public CG_RandomDispersalLayoutBase
{
   public:
  void userInitialize(LensContext* CG_c, int& total);
      ShallowArray< int > userExecute(LensContext* CG_c);
      RandomDispersalLayout();
      virtual ~RandomDispersalLayout();
      virtual void duplicate(std::unique_ptr<RandomDispersalLayout>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_RandomDispersalLayoutBase>&& dup) const;
      //   private:
      //      RNG _rng;
};

#endif
