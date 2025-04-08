#ifndef LoadMatrix_H
#define LoadMatrix_H

#include "Lens.h"
#include "CG_LoadMatrixBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>

class LoadMatrix : public CG_LoadMatrixBase
{
   public:
      void userInitialize(LensContext* CG_c, CustomString& filename, int& rows, int& cols);
      ShallowArray< float > userExecute(LensContext* CG_c);
      LoadMatrix();
      virtual ~LoadMatrix();
      virtual void duplicate(std::unique_ptr<LoadMatrix>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_LoadMatrixBase>&& dup) const;
};

#endif
