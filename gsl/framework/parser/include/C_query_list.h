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

#ifndef C_query_list_H
#define C_query_list_H
#include "Copyright.h"

#include <list>
#include <string>
#include "C_production.h"

class C_query;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
