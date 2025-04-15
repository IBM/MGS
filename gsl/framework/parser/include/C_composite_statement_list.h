// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
