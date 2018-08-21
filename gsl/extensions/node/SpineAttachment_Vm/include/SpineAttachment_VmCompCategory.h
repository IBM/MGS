// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef SpineAttachment_VmCompCategory_H
#define SpineAttachment_VmCompCategory_H

#include "Lens.h"
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
