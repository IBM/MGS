// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelKIR_STR_MSN_mouseCompCategory_H
#define ChannelKIR_STR_MSN_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelKIR_STR_MSN_mouseCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class ChannelKIR_STR_MSN_mouseCompCategory : public CG_ChannelKIR_STR_MSN_mouseCompCategory, public CountableModel
{
   public:
      ChannelKIR_STR_MSN_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
