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

#include "C_index_entry.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"


void C_index_entry::internalExecute(LensContext *c)
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
