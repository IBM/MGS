// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_type_definition_H
#define C_type_definition_H
#include "Copyright.h"

#include <memory>
#include <map>
#include "C_production.h"

class C_declarator;
class C_grid_definition_body;
class C_composite_definition_body;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
