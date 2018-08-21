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

#ifndef _C_FUNCTOR_DEFINITION_H
#define _C_FUNCTOR_DEFINITION_H
#include "Copyright.h"

#include <string>
#include <list>
#include "C_production.h"

class C_functor_category;
class C_declarator;
class C_parameter_type;
class C_parameter_type_list;
class C_complex_functor_definition;
class C_connection_script_definition;
class LensContext;
class FunctorType;
class ScriptFunctorType;
class SyntaxError;

class C_functor_definition : public C_production
{
   public:
      enum Basis { _BASIC, _CONSTR_DEF, _COMPLEX, _SCRIPT};
      C_functor_definition(const C_functor_definition&);
      C_functor_definition(C_functor_category *, C_declarator *, 
			   SyntaxError *);
      C_functor_definition(C_functor_category *, C_declarator *, 
			   C_parameter_type_list *, SyntaxError *);
      C_functor_definition(C_complex_functor_definition *, SyntaxError *);
      C_functor_definition(C_connection_script_definition *, SyntaxError *);
      virtual ~C_functor_definition();
      virtual C_functor_definition* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessor methods
      std::string getDeclarator(){
	 return _declaratorName;
      }

      std::string getCategory() {
	 return _categoryString;
      }

      std::list<C_parameter_type>* getConstructorParams() {
	 return _constructor_list;
      }

      std::list<C_parameter_type>* getFunctionParams() {
	 return _function_list;
      }

      std::list<C_parameter_type>* getReturnParams() {
	 return _return_list;
      }

      bool isScript()const {
	 return _basis==_SCRIPT;
      }

      FunctorType* getFunctorType() {
	 return _functorType;
      }

   private:
      Basis _basis;
      C_functor_category* _functor_category;
      std::string _categoryString;
      std::string _declaratorName;
      C_declarator* _declarator;
      C_parameter_type_list* _constructor_ptl;
      C_complex_functor_definition* _complex_functor_def;
      C_connection_script_definition* _c_script_def;
      FunctorType* _functorType;
      ScriptFunctorType* _sft;
      std::list<C_parameter_type> _empty;
      std::list<C_parameter_type>* _constructor_list;
      std::list<C_parameter_type>* _function_list;
      std::list<C_parameter_type>* _return_list;

      void basicWork(LensContext *c);
      void constrDefWork(LensContext *c);
      void complexWork(LensContext *c);
      void scriptWork(LensContext *c);
};
#endif
