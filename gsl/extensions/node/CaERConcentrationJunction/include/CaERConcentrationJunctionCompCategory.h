#ifndef CaERConcentrationJunctionCompCategory_H
#define CaERConcentrationJunctionCompCategory_H

#include "Lens.h"
#include "CG_CaERConcentrationJunctionCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class CaERConcentrationJunctionCompCategory
    : public CG_CaERConcentrationJunctionCompCategory,
      public CountableModel
{
  public:
  CaERConcentrationJunctionCompCategory(Simulation& sim,
                                        const std::string& modelName,
                                        const NDPairList& ndpList);
  void deriveParameters(RNG& rng);
  void count();
};

#endif
