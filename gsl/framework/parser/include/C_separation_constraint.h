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

#ifndef C_separation_constraint_H
#define C_separation_constraint_H
#include "Copyright.h"

#include <string>

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_declarator;
class C_nodeset;
class NodeSet;
class Variable;

class C_separation_constraint : public C_production
{
   public:
      C_separation_constraint(C_declarator *, SyntaxError *);
      C_separation_constraint(C_nodeset *, SyntaxError *);
      C_separation_constraint(const C_separation_constraint&);
      virtual ~C_separation_constraint();
      virtual C_separation_constraint* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      NodeSet* getNodeSetConstraint() const {
	 return _nodesetConstraint;
      }
      Variable* getVariableConstraint() const {
	 return _variableConstraint;
      }

   private:
      C_declarator* _declarator;
      C_nodeset* _nodeset;
      // not owned
      NodeSet* _nodesetConstraint;
      Variable* _variableConstraint;
      
};
#endif
