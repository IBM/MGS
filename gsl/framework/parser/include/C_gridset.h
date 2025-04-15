// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _C_GRIDSET_H_
#define _C_GRIDSET_H_
#include "Copyright.h"

#include "C_production.h"

class C_repname;
class C_gridnodeset;
class LensContext;
class GridSet;
class SyntaxError;

class C_gridset : public C_production
{
   public:
      C_gridset(const C_gridset&);
      C_gridset(C_repname *, C_gridnodeset *, SyntaxError *);
      virtual ~C_gridset();
      virtual C_gridset* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      GridSet* getGridSet() {
	 return _gridset;
      }

   private:
      C_repname* _repname;
      C_gridnodeset* _gridnodeset;
      GridSet* _gridset;

};
#endif
