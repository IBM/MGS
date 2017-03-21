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
