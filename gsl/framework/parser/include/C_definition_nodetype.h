// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_definition_nodetype_H
#define C_definition_nodetype_H
#include "Copyright.h"

#include "C_definition.h"

class C_declarator;
class GslContext;
class C_argument;
class SyntaxError;
class C_phase_mapping_list;

class C_definition_nodetype : public C_definition
{
   public:
      C_definition_nodetype(const C_definition_nodetype&);
      C_definition_nodetype(C_declarator *, C_argument *, 
			    C_phase_mapping_list *, SyntaxError *);
      virtual C_definition_nodetype* duplicate() const;
      virtual ~C_definition_nodetype ();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      C_declarator* getName() const { 
	 return _declarator; 
      }
      C_argument* getArgument() const { 
	 return _argument; 
      }

   private:
      C_declarator* _declarator;
      C_argument* _argument;
      C_phase_mapping_list* _phase_mapping_list;
};
#endif
