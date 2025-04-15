// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_connection_script_definition_body_H
#define C_connection_script_definition_body_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_connection_script_declaration;
class LensContext;
class DataItem;
class SyntaxError;

class C_connection_script_definition_body : public C_production
{
   public:
      C_connection_script_definition_body(
	 const C_connection_script_definition_body&);
      C_connection_script_definition_body(
	 C_connection_script_declaration *, SyntaxError *);
      C_connection_script_definition_body(
	 C_connection_script_definition_body *, 
	 C_connection_script_declaration *, SyntaxError *);
      std::list<C_connection_script_declaration*>* releaseList();
      virtual ~C_connection_script_definition_body ();
      virtual C_connection_script_definition_body* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const DataItem * getRVal() {
	 return _rval;
      }
      void setTdError(SyntaxError *tdError) { 
	 _tdError = tdError; 
      }
      void printTdError();

   private:
      std::list<C_connection_script_declaration*>* _list;
      const DataItem* _rval;
      SyntaxError* _tdError;
};
#endif
