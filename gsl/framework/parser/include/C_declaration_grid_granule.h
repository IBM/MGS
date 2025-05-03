// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_grid_granule_H
#define C_declaration_grid_granule_H
#include "Copyright.h"

#include "C_declaration.h"
#include "GslContext.h"
#include <string>
#include <memory>
#include <map>

class C_declarator;
class C_grid_granule;
class SyntaxError;
class GranuleMapper;

class C_declaration_grid_granule : public C_declaration
{
   public:
      C_declaration_grid_granule(const C_declaration_grid_granule&);
      C_declaration_grid_granule(C_declarator *, C_grid_granule *, SyntaxError *);
      virtual C_declaration_grid_granule* duplicate() const;
      virtual ~C_declaration_grid_granule();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_declarator* _declarator;
      C_grid_granule* _granuleMapper;
      std::string* _name;
};
#endif
