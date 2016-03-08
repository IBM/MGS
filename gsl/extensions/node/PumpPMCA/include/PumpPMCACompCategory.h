#ifndef PumpPMCACompCategory_H
#define PumpPMCACompCategory_H

#include "Lens.h"
#include "CG_PumpPMCACompCategory.h"
#include "CountableModel.h"

class NDPairList;

class PumpPMCACompCategory : public CG_PumpPMCACompCategory,
                             public CountableModel
{
  public:
  PumpPMCACompCategory(Simulation& sim, const std::string& modelName,
                       const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
