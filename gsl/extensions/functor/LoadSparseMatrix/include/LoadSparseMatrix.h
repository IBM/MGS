#ifndef LoadSparseMatrix_H
#define LoadSparseMatrix_H

#include "Lens.h"
#include "CG_LoadSparseMatrixBase.h"
#include "LensContext.h"
#include "ShallowArray.h"
#include <memory>

class LoadSparseMatrix : public CG_LoadSparseMatrixBase
{
   public:
      void userInitialize(LensContext* CG_c, CustomString& filepath, CustomString& filename);
      ShallowArray<float> userExecute(LensContext* CG_c);
      LoadSparseMatrix();
      virtual ~LoadSparseMatrix();
      virtual void duplicate(std::unique_ptr<LoadSparseMatrix>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_LoadSparseMatrixBase>&& dup) const;
};

#endif
