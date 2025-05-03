// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_type_definition_H
#define C_type_definition_H
#include "Copyright.h"

#include <memory>
#include <map>
#include "C_production.h"

class C_declarator;
class C_grid_definition_body;
class C_composite_definition_body;
class GslContext;
class SyntaxError;

class C_type_definition : public C_production
{
   public:
      C_type_definition(const C_type_definition&);
      C_type_definition(C_declarator *, C_grid_definition_body *, 
			SyntaxError *);
      C_type_definition(C_declarator *, C_grid_definition_body *, 
			C_declarator *, SyntaxError *);
      C_type_definition(C_declarator *, C_composite_definition_body *, 
			SyntaxError *);
      C_type_definition(C_declarator *, C_composite_definition_body *,  
			C_declarator *, SyntaxError *);
      C_type_definition(SyntaxError *);
      virtual ~C_type_definition();
      virtual C_type_definition* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
   private:
      enum Type{_GRID, _COMPOSITE};
      bool _declaration;
      C_declarator* _typeName;
      C_declarator* _instanceName;
      C_grid_definition_body* _gridDefBody;
      C_composite_definition_body* _compositeDefBody;
      Type _type;

};
#endif
