// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef BackwardSolvePoint4CompCategory_H
#define BackwardSolvePoint4CompCategory_H

#include "Lens.h"
#include "CG_BackwardSolvePoint4CompCategory.h"

class NDPairList;

class BackwardSolvePoint4CompCategory : public CG_BackwardSolvePoint4CompCategory
{
   public:
      BackwardSolvePoint4CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
