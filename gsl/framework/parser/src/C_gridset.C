// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_gridset.h"
#include "C_repname.h"
#include "C_gridnodeset.h"
#include "GridSet.h"
#include "Repertoire.h"
#include "VectorOstream.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"


void C_gridset::internalExecute(GslContext *c)
{
   _repname->execute(c);
   _gridnodeset->execute(c);

   if (_repname->getRepertoire() == 0) {
      std::string mes = "repertoire is 0";
      throwError(mes);
   }
   if (_repname->getRepertoire()->getGrid() == 0) {
      std::string mes = "grid is 0";
      throwError(mes);
   }
   _gridset = new GridSet(_repname->getRepertoire()->getGrid());
   const std::vector<int>& begin = _gridnodeset->getBeginCoords(), 
      increment = _gridnodeset->getIncrement(), end = _gridnodeset->getEndCoords();
   if (begin.size() != 0 && increment.size()!= 0 && end.size()!= 0) {
      _gridset->setCoords(begin, increment, end);
   }
}

C_gridset::C_gridset(const C_gridset& rv) 
   : C_production(rv), _repname(0), _gridnodeset(0), _gridset(0)
{
   if (rv._repname) {
      _repname = rv._repname->duplicate();
   }
   if (rv._gridnodeset) {
      _gridnodeset = rv._gridnodeset->duplicate();
   }
   if (rv._gridset) {
      _gridset = new GridSet(*rv._gridset);
   }
}


C_gridset::C_gridset(C_repname* r, C_gridnodeset* g, SyntaxError* error)
   : C_production(error), _repname(r), _gridnodeset(g), _gridset(0)
{
}


C_gridset* C_gridset::duplicate() const
{
   return new C_gridset(*this);
}


C_gridset::~C_gridset()
{
   delete _repname;
   delete _gridnodeset;
   delete _gridset;
}

void C_gridset::checkChildren() 
{
   if (_repname) {
      _repname->checkChildren();
      if (_repname->isError()) {
         setError();
      }
   }
   if (_gridnodeset) {
      _gridnodeset->checkChildren();
      if (_gridnodeset->isError()) {
         setError();
      }
   }
} 

void C_gridset::recursivePrint() 
{
   if (_repname) {
      _repname->recursivePrint();
   }
   if (_gridnodeset) {
      _gridnodeset->recursivePrint();
   }
   printErrorMessage();
} 
