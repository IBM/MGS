// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_GRID_GRANULE_H
#define C_GRID_GRANULE_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_production.h"
#include <vector>

class GslContext;
class GranuleMapper;
class ConnectionIncrement;
class C_declarator;

class C_grid_granule : public C_production
{
   public:
      C_grid_granule(C_declarator*, SyntaxError* error);
      C_grid_granule(const C_grid_granule&);
      virtual ~C_grid_granule();
      virtual C_grid_granule* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual GranuleMapper* getGranuleMapper() {return _granuleMapper;}

   private:
      C_declarator* _declarator;
      GranuleMapper* _granuleMapper;
};
#endif
