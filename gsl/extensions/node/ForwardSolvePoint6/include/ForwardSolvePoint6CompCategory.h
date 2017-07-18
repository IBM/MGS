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

#ifndef ForwardSolvePoint6CompCategory_H
#define ForwardSolvePoint6CompCategory_H

#include "Lens.h"
#include "CG_ForwardSolvePoint6CompCategory.h"

class NDPairList;

class ForwardSolvePoint6CompCategory : public CG_ForwardSolvePoint6CompCategory
{
   public:
      ForwardSolvePoint6CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
