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

#ifndef BackwardSolvePoint3CompCategory_H
#define BackwardSolvePoint3CompCategory_H

#include "Lens.h"
#include "CG_BackwardSolvePoint3CompCategory.h"

class NDPairList;

class BackwardSolvePoint3CompCategory : public CG_BackwardSolvePoint3CompCategory
{
   public:
      BackwardSolvePoint3CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
