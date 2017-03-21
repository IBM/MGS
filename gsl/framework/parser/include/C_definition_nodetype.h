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

#ifndef C_definition_nodetype_H
#define C_definition_nodetype_H
#include "Copyright.h"

#include "C_definition.h"

class C_declarator;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
