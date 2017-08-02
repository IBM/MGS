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

#ifndef C_complex_functor_declaration_body_H
#define C_complex_functor_declaration_body_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_constructor_clause;
class C_function_clause;
class C_parameter_type;
class C_return_clause;
class LensContext;
class C_complex_functor_clause_list;
class SyntaxError;

class C_complex_functor_declaration_body : public C_production
{
   public:
      C_complex_functor_declaration_body(
	 const C_complex_functor_declaration_body&);
      C_complex_functor_declaration_body(C_complex_functor_clause_list *, 
					 SyntaxError *);
      C_complex_functor_declaration_body(
	 C_constructor_clause *, C_function_clause *, C_return_clause *, 
	 SyntaxError *);
      virtual ~C_complex_functor_declaration_body();
      virtual C_complex_functor_declaration_body* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_parameter_type>* getConstructorParameters() { 
	 return _constructor_ptl; 
      }
      std::list<C_parameter_type>* getFunctionParameters() { 
	    return _function_ptl; 
      }
      std::list<C_parameter_type>* getReturnParameters() { 
	 return _return_ptl; 
      }

   private:
      std::list<C_parameter_type>* _constructor_ptl;
      std::list<C_parameter_type>* _function_ptl;
      std::list<C_parameter_type>* _return_ptl;
      std::list<C_parameter_type> _empty;

      C_constructor_clause* _constructorClause;
      C_function_clause* _functionClause;
      C_return_clause* _returnClause;
      C_complex_functor_clause_list* _complexFunctorClauseList;
      bool _ownPointers;
};
#endif
