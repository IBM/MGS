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

#ifndef ForwardSolvePoint3CompCategory_H
#define ForwardSolvePoint3CompCategory_H

#include "Lens.h"
#include "CG_ForwardSolvePoint3CompCategory.h"

class NDPairList;

class ForwardSolvePoint3CompCategory : public CG_ForwardSolvePoint3CompCategory
{
   public:
      ForwardSolvePoint3CompCategory(Simulation& sim, const std::string& modelName, const NDPairList& ndpList);
};

#endif
