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

#ifndef SpineAttachment_VmCaiCompCategory_H
#define SpineAttachment_VmCaiCompCategory_H

#include "Lens.h"
#include "CG_SpineAttachment_VmCaiCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class SpineAttachment_VmCaiCompCategory : public CG_SpineAttachment_VmCaiCompCategory, public CountableModel
{
   public:
      SpineAttachment_VmCaiCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
