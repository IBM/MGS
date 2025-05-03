// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_function_clause_H
#define C_function_clause_H
#include "Copyright.h"

#include <list>
#include "C_complex_functor_clause.h"

class C_parameter_type;
class C_parameter_type_list;
class GslContext;
class SyntaxError;

class C_function_clause : public C_complex_functor_clause
{
   public:
      C_function_clause(const C_function_clause&);
      C_function_clause(C_parameter_type_list *, SyntaxError *);
      virtual ~C_function_clause();
      virtual C_function_clause* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_parameter_type>* getParameterTypeList();
   private:
      C_parameter_type_list* _ptl;
};
#endif
