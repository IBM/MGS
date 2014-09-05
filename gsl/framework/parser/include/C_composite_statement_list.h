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

#ifndef C_composite_statement_list_H
#define C_composite_statement_list_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_composite_statement;
class C_composite_statement_list;
class LensContext;
class SyntaxError;

class C_composite_statement_list : public C_production
{
   public:
      C_composite_statement_list(const C_composite_statement_list&);
      C_composite_statement_list(C_composite_statement *, SyntaxError *);
      C_composite_statement_list(C_composite_statement_list *,  
				 C_composite_statement *, SyntaxError *);
      std::list<C_composite_statement*>* releaseList();
      virtual ~C_composite_statement_list();
      virtual C_composite_statement_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<C_composite_statement*>* getListCompositeStatement() const { 
	 return _list; 
      }
      void setTdError(SyntaxError *tdError) {
	 _tdError = tdError; 
      }
      void printTdError();

   private:
      std::list<C_composite_statement*>* _list;
      SyntaxError* _tdError;
};
#endif
