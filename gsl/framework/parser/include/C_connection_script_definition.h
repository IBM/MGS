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

#ifndef C_connection_script_definition_H
#define C_connection_script_definition_H
#include "Copyright.h"

#include <list>
#include <string>
#include "C_production.h"

class C_declarator;
class C_parameter_type_list;
class C_parameter_type;
class C_connection_script_definition_body;
class LensContext;
class Functor;
class SyntaxError;

class C_connection_script_definition : public C_production
{
   public:
      C_connection_script_definition(const C_connection_script_definition&);
      C_connection_script_definition(C_declarator *, C_parameter_type_list *, 
	 C_connection_script_definition_body *, SyntaxError *);
      virtual ~C_connection_script_definition();
      virtual C_connection_script_definition* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Functor* getFunctor() {
	 return _functor;
      }

      std::string getScriptName() const {
	 return _scriptName;
      }

      const std::string& getName();
      std::list<C_parameter_type>* getFunctionParameters();

   private:
      C_declarator* _declarator;
      C_parameter_type_list* _param_type_list;
      C_connection_script_definition_body* _script_body;
      Functor* _functor;
      std::string _scriptName;
      // Only for an empty name, real name is from _declarator
      std::string _name; 
};
#endif
