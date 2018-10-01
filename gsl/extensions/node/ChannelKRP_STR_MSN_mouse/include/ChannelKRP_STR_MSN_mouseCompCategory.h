// =================================================================
//
// (C) Copyright 2018 New Jersey Institute of Technology.
//
// =================================================================


#ifndef ChannelKRP_STR_MSN_mouseCompCategory_H
#define ChannelKRP_STR_MSN_mouseCompCategory_H

#include "Lens.h"
#include "CG_ChannelKRP_STR_MSN_mouseCompCategory.h"
#include "CountableModel.h" 

class NDPairList;

class ChannelKRP_STR_MSN_mouseCompCategory : public CG_ChannelKRP_STR_MSN_mouseCompCategory, public CountableModel
{
   public:
      ChannelKRP_STR_MSN_mouseCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void computeE(RNG& rng);
      void count();
};

#endif
