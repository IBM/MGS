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

#ifndef Connexon_VmCaiCompCategory_H
#define Connexon_VmCaiCompCategory_H

#include "Lens.h"
#include "CG_Connexon_VmCaiCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class Connexon_VmCaiCompCategory : public CG_Connexon_VmCaiCompCategory, public CountableModel
{
   public:
      Connexon_VmCaiCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
