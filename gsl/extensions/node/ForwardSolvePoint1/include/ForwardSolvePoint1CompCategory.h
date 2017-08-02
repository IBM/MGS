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

#ifndef ForwardSolvePoint1CompCategory_H
#define ForwardSolvePoint1CompCategory_H

#include "Lens.h"
#include "CG_ForwardSolvePoint1CompCategory.h"

class NDPairList;

class ForwardSolvePoint1CompCategory : public CG_ForwardSolvePoint1CompCategory
{
   public:
      ForwardSolvePoint1CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
