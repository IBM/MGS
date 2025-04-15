// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_complex_functor_clause_list_H
#define C_complex_functor_clause_list_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_complex_functor_clause;
class LensContext;
class C_complex_functor_clause;
class SyntaxError;

class C_complex_functor_clause_list : public C_production
{
   public:
      C_complex_functor_clause_list(const C_complex_functor_clause_list&);
      C_complex_functor_clause_list(C_complex_functor_clause *, SyntaxError *);
      C_complex_functor_clause_list(C_complex_functor_clause_list *, 
				    C_complex_functor_clause *, SyntaxError *);
      std::list<C_complex_functor_clause*>* releaseList();
      virtual ~C_complex_functor_clause_list();
      virtual C_complex_functor_clause_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_complex_functor_clause*>* getList() {
	 return _list;
      }

   private:
      std::list<C_complex_functor_clause*>* _list;
};
#endif
