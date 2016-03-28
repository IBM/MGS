// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
