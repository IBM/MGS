#ifndef mGluReceptorType1CompCategory_H
#define mGluReceptorType1CompCategory_H

#include "Lens.h"
#include "CG_mGluReceptorType1CompCategory.h"
#include "CountableModel.h"

class NDPairList;

class mGluReceptorType1CompCategory : public CG_mGluReceptorType1CompCategory,
                                 public CountableModel
{
   public:
      mGluReceptorType1CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      //void computeTadj(RNG& rng);
      void count();
};

#endif
