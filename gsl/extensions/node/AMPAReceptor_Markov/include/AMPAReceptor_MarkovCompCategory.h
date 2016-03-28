#ifndef AMPAReceptor_MarkovCompCategory_H
#define AMPAReceptor_MarkovCompCategory_H

#include "Lens.h"
#include "CG_AMPAReceptor_MarkovCompCategory.h"
#include "CountableModel.h"  //new

class NDPairList;

class AMPAReceptor_MarkovCompCategory
    : public CG_AMPAReceptor_MarkovCompCategory,
      public CountableModel
{
  public:
  AMPAReceptor_MarkovCompCategory(Simulation& sim, const std::string& modelName,
                                  const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
