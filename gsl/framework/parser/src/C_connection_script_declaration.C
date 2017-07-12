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

#include "C_connection_script_declaration.h"
#include "C_declaration.h"
#include "C_directive.h"
#include "SyntaxError.h"
#include "C_production.h"

const DataItem* C_connection_script_declaration::getRVal() const
{
   DataItem const* retval = 0;
   if (_return) {
      retval = _directive->getRVal();
   }
   return retval;
}

void C_connection_script_declaration::internalExecute(LensContext *c)
{
   if (_directive) _directive->execute(c);
   else _declaration->execute(c);
}


C_connection_script_declaration::C_connection_script_declaration(
   const C_connection_script_declaration& rv)
   : C_production(rv), _declaration(0), _directive(0), _return(rv._return)
{
   if (rv._declaration) {
      _declaration = rv._declaration->duplicate();
   }
   if (rv._directive) {
      _directive = rv._directive->duplicate();
   }
}


C_connection_script_declaration::C_connection_script_declaration(
   C_declaration *d, SyntaxError * error)
   : C_production(error), _declaration(d), _directive(0), _return(false)
{
}


C_connection_script_declaration::C_connection_script_declaration(
   bool b, C_directive *d, SyntaxError * error)
   : C_production(error), _declaration(0), _directive(d), _return(b)
{
}


C_connection_script_declaration* 
C_connection_script_declaration::duplicate() const
{
   return new C_connection_script_declaration(*this);
}


C_connection_script_declaration::~C_connection_script_declaration()
{
   delete _directive;
   delete _declaration;
}

void C_connection_script_declaration::checkChildren() 
{
   if (_declaration) {
      _declaration->checkChildren();
      if (_declaration->isError()) {
         setError();
      }
   }
   if (_directive) {
      _directive->checkChildren();
      if (_directive->isError()) {
         setError();
      }
   }
} 

void C_connection_script_declaration::recursivePrint() 
{
   if (_declaration) {
      _declaration->recursivePrint();
   }
   if (_directive) {
      _directive->recursivePrint();
   }
   printErrorMessage();
} 
