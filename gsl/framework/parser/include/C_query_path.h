// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_query_path_H
#define C_query_path_H
#include "Copyright.h"

#include <list>
#include <vector>
#include <string>
#include "C_production.h"

class C_query_list;
class C_repname;
class GslContext;
class Publisher;
class QueryField;
class SyntaxError;

class C_query_path : public C_production
{
   public:
      C_query_path(const C_query_path&);
      C_query_path(SyntaxError *);
      C_query_path(C_query_list *, SyntaxError *);
      C_query_path(C_repname *, C_query_list *, SyntaxError *);
      virtual C_query_path* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual ~C_query_path ();
      Publisher* getPublisher() {return _publisher;}

   private:
      void setField(const std::vector<QueryField*>& fields, 
		    const std::string& name, const std::string& entry, 
		    int nbrEntries);
      C_query_list* _queryList;
      C_repname* _repName;
      Publisher* _publisher;
};
#endif
