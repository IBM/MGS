// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================

#ifndef GetPreNodeIndex_H
#define GetPreNodeIndex_H

#include "Mgs.h"
#include "CG_GetPreNodeIndexBase.h"
#include "LensContext.h"
#include <memory>

class GetPreNodeIndex : public CG_GetPreNodeIndexBase
{
   public:
      void userInitialize(LensContext* CG_c);
      int userExecute(LensContext* CG_c);
      GetPreNodeIndex();
      virtual ~GetPreNodeIndex();
      virtual void duplicate(std::unique_ptr<GetPreNodeIndex>&& dup) const;
      virtual void duplicate(std::unique_ptr<Functor>&& dup) const;
      virtual void duplicate(std::unique_ptr<CG_GetPreNodeIndexBase>&& dup) const;
};

#endif
