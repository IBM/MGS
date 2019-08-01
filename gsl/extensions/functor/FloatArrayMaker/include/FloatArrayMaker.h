#ifndef FloatArrayMaker_H
#define FloatArrayMaker_H

#include "Lens.h"
#include "CG_FloatArrayMakerBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>

class FloatArrayMaker : public CG_FloatArrayMakerBase
{
   public:
      void userInitialize(LensContext* CG_c, Functor*& f, int& size);
      ShallowArray< float > userExecute(LensContext* CG_c);
      FloatArrayMaker();
      virtual ~FloatArrayMaker();
      virtual void duplicate(std::unique_ptr<FloatArrayMaker>& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>& dup) const;
      virtual void duplicate(std::unique_ptr<CG_FloatArrayMakerBase>& dup) const;
};

#endif
