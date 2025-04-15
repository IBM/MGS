// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SpineAttachment_VmCaiCaERCompCategory_H
#define SpineAttachment_VmCaiCaERCompCategory_H

#include "Lens.h"
#include "CG_SpineAttachment_VmCaiCaERCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SpineAttachment_VmCaiCaERCompCategory
    : public CG_SpineAttachment_VmCaiCaERCompCategory,
      public CountableModel
{
  public:
  SpineAttachment_VmCaiCaERCompCategory(Simulation& sim,
                                        const std::string& modelName,
                                        const NDPairList& ndpList);
  void count();
};

#endif
