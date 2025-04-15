// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_declaration_separation_constraint_H
#define C_declaration_separation_constraint_H 
#include "Copyright.h"

#include <vector>

#include "C_declaration.h"

class LensContext;
class DataItem;
class C_separation_constraint;
class C_separation_constraint_list;

class C_declaration_separation_constraint : public C_declaration
{
   public:
      C_declaration_separation_constraint(
	 const C_declaration_separation_constraint&);
      C_declaration_separation_constraint(C_separation_constraint_list *, 
					  SyntaxError *);
      virtual ~C_declaration_separation_constraint();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual C_declaration* duplicate() const;

   private:
      C_separation_constraint_list* _separationConstraintList;
};
#endif
