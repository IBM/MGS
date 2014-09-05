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

#ifndef C_complex_functor_definition_H
#define C_complex_functor_definition_H
#include "Copyright.h"

#include <list>
#include <string>
#include "C_production.h"

class C_functor_category;
class C_declarator;
class C_complex_functor_declaration_body;
class C_parameter_type;
class LensContext;
class SyntaxError;

class C_complex_functor_definition : public C_production
{
   public:
      C_complex_functor_definition(const C_complex_functor_definition&);
      C_complex_functor_definition(C_functor_category *, C_declarator *, 
				   C_complex_functor_declaration_body *, 
				   SyntaxError *);
      virtual ~C_complex_functor_definition();
      virtual C_complex_functor_definition* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::string& getCategory() const;
      const std::string& getName();
      std::list<C_parameter_type>* getConstructorParameters();
      std::list<C_parameter_type>* getFunctionParameters();
      std::list<C_parameter_type>* getReturnParameters();

   private:
      C_functor_category* _functorCategory;
      C_declarator* _declarator;
      C_complex_functor_declaration_body* _complexFunctorDec;
      // Only for an empty name, real name is from _declarator
      std::string _name;	
};
#endif
