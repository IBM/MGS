// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _C_NDPAIR_CLAUSE_LIST_BODY_H_
#define _C_NDPAIR_CLAUSE_LIST_BODY_H_
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_ndpair_clause;
class GslContext;
class NDPairList;
class SyntaxError;

class C_ndpair_clause_list_body : public C_production
{
   public:
      C_ndpair_clause_list_body(const C_ndpair_clause_list_body&);
      C_ndpair_clause_list_body(C_ndpair_clause *, SyntaxError *);
      C_ndpair_clause_list_body(C_ndpair_clause *, 
				C_ndpair_clause_list_body *, SyntaxError *);
      virtual ~C_ndpair_clause_list_body ();
      virtual C_ndpair_clause_list_body* duplicate() const;
      std::list<C_ndpair_clause*>* releaseList();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      NDPairList* getNDPairList(){
	 return _ndpairList;
      }

   private:
      NDPairList* _ndpairList;
      std::list<C_ndpair_clause*>* _list;
};
#endif
