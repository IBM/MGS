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

#ifndef _C_NDPAIR_CLAUSE_LIST_H
#define _C_NDPAIR_CLAUSE_LIST_H
#include "Copyright.h"

#include <list>
#include <memory>
#include "C_production.h"

class NDPairList;
class C_ndpair_clause_list_body;
class LensContext;
class SyntaxError;

class C_ndpair_clause_list : public C_production
{
   public:
      C_ndpair_clause_list(const C_ndpair_clause_list&);
      C_ndpair_clause_list(C_ndpair_clause_list_body *, SyntaxError *);
      C_ndpair_clause_list(C_ndpair_clause_list *, 
			   C_ndpair_clause_list_body *, SyntaxError *);
      C_ndpair_clause_list(SyntaxError *);
      virtual ~C_ndpair_clause_list ();
      virtual C_ndpair_clause_list* duplicate() const;
      std::list<C_ndpair_clause_list_body*> * releaseList();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      NDPairList* getList() {
	 return _list;
      }
      void releaseList(std::auto_ptr<NDPairList>& ndp);
      const NDPairList& getNDPList() {
	 return *_list;
      }

   private:
      std::list<C_ndpair_clause_list_body*>* _bodyList;
      NDPairList* _list;
};
#endif
