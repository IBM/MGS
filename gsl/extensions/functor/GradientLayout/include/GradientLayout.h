#ifndef GradientLayout_H
#define GradientLayout_H

#include "Lens.h"
#include "CG_GradientLayoutBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>

class GradientLayout : public CG_GradientLayoutBase
{
   public:
      void userInitialize(LensContext* CG_c, int& total, double& slope, ShallowArray< int >& origin, int& originDensity, ShallowArray< bool >& gradientDimensions);
      ShallowArray< int > userExecute(LensContext* CG_c);
      GradientLayout();
      virtual ~GradientLayout();
      virtual void duplicate(std::auto_ptr<GradientLayout>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_GradientLayoutBase>& dup) const;
};

#endif
