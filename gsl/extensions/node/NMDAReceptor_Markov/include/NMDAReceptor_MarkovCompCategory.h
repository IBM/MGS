#ifndef NMDAReceptor_MarkovCompCategory_H
#define NMDAReceptor_MarkovCompCategory_H

#include "CG_NMDAReceptor_MarkovCompCategory.h"
#include "CountableModel.h"
#include "Lens.h"

class NDPairList;

class NMDAReceptor_MarkovCompCategory
    : public CG_NMDAReceptor_MarkovCompCategory,
      public CountableModel
{
  public:
  NMDAReceptor_MarkovCompCategory(Simulation& sim, const std::string& modelName,
                                  const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
