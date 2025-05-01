// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SpineAttachment_VmCompCategory_H
#define SpineAttachment_VmCompCategory_H

#include "Mgs.h"
#include "CG_SpineAttachment_VmCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SpineAttachment_VmCompCategory : public CG_SpineAttachment_VmCompCategory,
                                       public CountableModel
{
  public:
  SpineAttachment_VmCompCategory(Simulation& sim, const std::string& modelName,
                                 const NDPairList& ndpList);
  void count();
};

#endif
