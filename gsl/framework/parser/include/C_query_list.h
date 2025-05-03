// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_query_list_H
#define C_query_list_H
#include "Copyright.h"

#include <list>
#include <string>
#include "C_production.h"

class C_query;
class GslContext;
class SyntaxError;

class C_query_list : public C_production
{
   public:
      C_query_list(const C_query_list&);
      //C_query_list(SyntaxError *);
      C_query_list(C_query *, SyntaxError *);
      C_query_list(C_query_list *, C_query *, SyntaxError *);
      virtual C_query_list* duplicate() const;
      std::list<C_query>* releaseList();
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual ~C_query_list ();
      std::list<C_query>* getList() {
	 return _list;
      }

   private:
      std::list<C_query> *_list;
};
#endif
