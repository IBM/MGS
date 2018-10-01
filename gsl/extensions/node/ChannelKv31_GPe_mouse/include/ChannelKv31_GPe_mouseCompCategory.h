// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelKv31_GPe_mouseCompCategory_H
#define ChannelKv31_GPe_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelKv31_GPe_mouseCompCategory.h"

#include "CountableModel.h"
class NDPairList;

class ChannelKv31_GPe_mouseCompCategory : public CG_ChannelKv31_GPe_mouseCompCategory,
                                          public CountableModel
{
   public:
      ChannelKv31_GPe_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
