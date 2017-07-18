// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_DEFINITION_VARIABLETYPE_H
#define C_DEFINITION_VARIABLETYPE_H
#include "Copyright.h"

#include <string>
#include <list>

#include "C_definition.h"
#include "C_name.h"

class C_declarator;
class C_parameter_type;
class C_parameter_type_list;
class LensContext;
class InstanceFactory;
class SyntaxError;
class C_phase_mapping_list;

class C_definition_variabletype : public C_definition
{
   public:
      C_definition_variabletype(const C_definition_variabletype&);
      C_definition_variabletype(C_declarator *, C_phase_mapping_list *, 
				SyntaxError *);
      C_definition_variabletype(C_declarator *, C_name*, 
			  C_phase_mapping_list *, 
				SyntaxError *);
      virtual ~C_definition_variabletype();
      virtual C_definition_variabletype* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessor methods
      std::string getDeclarator();
      std::string getArgument();
      InstanceFactory* getInstanceFactory();

   private:
      C_declarator* _declarator;
      InstanceFactory* _instanceFactory;
      C_phase_mapping_list* _phase_mapping_list;
	  C_name* _info; // which should be interpreted properly
};
/*
 * HISTORY:
 * 2016-Apr-18: add C_name* _info
 *       VariableType SimulationInfo(global) { init->init2; update->init3; }
 *  with global is accepted as a C_name
 */

#endif
