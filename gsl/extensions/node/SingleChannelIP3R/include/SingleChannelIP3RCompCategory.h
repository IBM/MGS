#ifndef SingleChannelIP3RCompCategory_H
#define SingleChannelIP3RCompCategory_H

#include "Lens.h"
#include "CG_SingleChannelIP3RCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SingleChannelIP3RCompCategory : public CG_SingleChannelIP3RCompCategory,
  public CountableModel
{
   public:
      SingleChannelIP3RCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
};

#endif
