// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_index_entry.h"
#include "GslContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"


void C_index_entry::internalExecute(GslContext *c)
{
   if (_from > _to) {
      std::string mes = " inverted index entry : (from, to)";
      throwError(mes);
   }
}

C_index_entry::C_index_entry(const C_index_entry& rv)
   : C_production(rv), _from(rv._from), _increment(rv._increment), _to(rv._to)
{
}

C_index_entry::C_index_entry(int i, SyntaxError * error)
   : C_production(error), _from(i), _increment(1), _to(i)
{
}


C_index_entry::C_index_entry(int from, int to, SyntaxError * error)
   : C_production(error), _from(from), _increment(1), _to(to)
{
}

C_index_entry::C_index_entry(int from, int increment, int to, SyntaxError * error)
   : C_production(error), _from(from), _increment(increment), _to(to)
{
}

C_index_entry* C_index_entry::duplicate() const
{
   return new C_index_entry(*this);
}


C_index_entry::~C_index_entry()
{
}

void C_index_entry::checkChildren() 
{
} 

void C_index_entry::recursivePrint() 
{
   printErrorMessage();
} 
