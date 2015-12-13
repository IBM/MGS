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

#ifndef Connexon_VmCompCategory_H
#define Connexon_VmCompCategory_H

#include "Lens.h"
#include "CG_Connexon_VmCompCategory.h"
#include "CountableModel.h"

class NDPairList;

class Connexon_VmCompCategory : public CG_Connexon_VmCompCategory, public CountableModel
{
   public:
      Connexon_VmCompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
      void count();
};

#endif
