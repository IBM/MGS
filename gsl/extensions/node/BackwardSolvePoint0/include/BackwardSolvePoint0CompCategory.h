// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef BackwardSolvePoint0CompCategory_H
#define BackwardSolvePoint0CompCategory_H

#include "Lens.h"
#include "CG_BackwardSolvePoint0CompCategory.h"

class NDPairList;

class BackwardSolvePoint0CompCategory : public CG_BackwardSolvePoint0CompCategory
{
   public:
      BackwardSolvePoint0CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
