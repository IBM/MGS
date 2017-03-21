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

#ifndef ForwardSolvePoint2CompCategory_H
#define ForwardSolvePoint2CompCategory_H

#include "Lens.h"
#include "CG_ForwardSolvePoint2CompCategory.h"

class NDPairList;

class ForwardSolvePoint2CompCategory : public CG_ForwardSolvePoint2CompCategory
{
   public:
      ForwardSolvePoint2CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
