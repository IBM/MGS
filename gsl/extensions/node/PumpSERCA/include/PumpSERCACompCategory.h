#ifndef PumpSERCACompCategory_H
#define PumpSERCACompCategory_H

#include "Lens.h"
#include "CG_PumpSERCACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class PumpSERCACompCategory : public CG_PumpSERCACompCategory,
                              public CountableModel
{
  public:
  PumpSERCACompCategory(Simulation& sim, const std::string& modelName,
                        const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
