// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SpineAttachment_VmCaiCompCategory_H
#define SpineAttachment_VmCaiCompCategory_H

#include "Lens.h"
#include "CG_SpineAttachment_VmCaiCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SpineAttachment_VmCaiCompCategory
    : public CG_SpineAttachment_VmCaiCompCategory,
      public CountableModel
{
  public:
  SpineAttachment_VmCaiCompCategory(Simulation& sim,
                                    const std::string& modelName,
                                    const NDPairList& ndpList);
  void count();
};

#endif
