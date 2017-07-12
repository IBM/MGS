#ifndef ChannelIP3RCompCategory_H
#define ChannelIP3RCompCategory_H

#include "Lens.h"
#include "CG_ChannelIP3RCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelIP3RCompCategory : public CG_ChannelIP3RCompCategory, 
   public CountableModel
{
   public:
      ChannelIP3RCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeTadj(RNG& rng);
      void count();
};

#endif
