#ifndef CaExtrusionCompCategory_H
#define CaExtrusionCompCategory_H

#include "Lens.h"
#include "CG_CaExtrusionCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class CaExtrusionCompCategory : public CG_CaExtrusionCompCategory,
                                public CountableModel
{
  public:
  CaExtrusionCompCategory(Simulation& sim, const std::string& modelName,
                          const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
