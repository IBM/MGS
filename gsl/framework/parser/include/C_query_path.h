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

#ifndef C_query_path_H
#define C_query_path_H
#include "Copyright.h"

#include <list>
#include <vector>
#include <string>
#include "C_production.h"

class C_query_list;
class C_repname;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
