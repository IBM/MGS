// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef _C_NDPAIR_CLAUSE_LIST_BODY_H_
#define _C_NDPAIR_CLAUSE_LIST_BODY_H_
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_ndpair_clause;
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
