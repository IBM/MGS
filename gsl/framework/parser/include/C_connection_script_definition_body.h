// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
