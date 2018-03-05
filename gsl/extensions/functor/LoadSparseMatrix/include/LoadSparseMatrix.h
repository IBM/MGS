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
      void userInitialize(LensContext* CG_c, String& filename);
      ShallowArray<float> userExecute(LensContext* CG_c);
      LoadSparseMatrix();
      virtual ~LoadSparseMatrix();
      virtual void duplicate(std::auto_ptr<LoadSparseMatrix>& dup) const;
      virtual void duplicate(std::auto_ptr<Functor>& dup) const;
      virtual void duplicate(std::auto_ptr<CG_LoadSparseMatrixBase>& dup) const;
};

#endif
