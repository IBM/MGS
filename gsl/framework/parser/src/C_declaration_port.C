// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_declaration_port.h"
#include "LensContext.h"
#include "C_declarator.h"
#include "IntDataItem.h"
#include "Simulation.h"
#include "SyntaxError.h"
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include "SyntaxErrorException.h"

extern "C" { int CreateSocket(int *, int); }

void C_declaration_port::internalExecute(LensContext *c)
{
   _declarator->execute(c);
   int fd;
   if (!CreateSocket(&fd, _portValue))
      std::cerr << "Problem creating socket on port "
		<< _portValue << "!" << std::endl;
   c->sim->addSocket(fd);

   // transfer fd to DataItem
   IntDataItem *idi = new IntDataItem;
   idi->setInt(fd);
   std::unique_ptr<DataItem> idi_ap(idi);
   try {
      c->symTable.addEntry(_declarator->getName(), idi_ap);
   } catch (SyntaxErrorException& e) {
      throwError("While declaring port, " + e.getError());
   }
}


C_declaration_port* C_declaration_port::duplicate() const
{
   return new C_declaration_port(*this);
}


C_declaration_port::C_declaration_port(const C_declaration_port& rv)
   : C_declaration(rv), _declarator(0), _portValue(rv._portValue)
{
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
}


C_declaration_port::~C_declaration_port()
{
   delete _declarator;
}


C_declaration_port::C_declaration_port(C_declarator *d, int i, 
				       SyntaxError * error)
   : C_declaration(error), _declarator(d), _portValue(i)
{
}

void C_declaration_port::checkChildren() 
{
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_declaration_port::recursivePrint() 
{
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
