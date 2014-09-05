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
