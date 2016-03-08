#ifndef ExchangerNCXCompCategory_H
#define ExchangerNCXCompCategory_H

#include "Lens.h"
#include "CG_ExchangerNCXCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ExchangerNCXCompCategory : public CG_ExchangerNCXCompCategory,
                                 public CountableModel
{
  public:
  ExchangerNCXCompCategory(Simulation& sim, const std::string& modelName,
                           const NDPairList& ndpList);
  void computeTadj(RNG& rng);
  void count();
};

#endif
