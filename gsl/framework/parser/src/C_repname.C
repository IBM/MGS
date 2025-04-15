// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_repname.h"
#include "C_preamble.h"
#include "Repertoire.h"
#include "LensContext.h"
#include "RepertoireDataItem.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

void C_repname::internalExecute(LensContext *c)
{
   _path.clear();
   if(_preamble) {
      _preamble->execute(c);
      _path = *_preamble->getListStrings();
   }
   _path.push_back(*_name);

   std::list<std::string>::const_iterator i, begin = _path.begin();
   std::list<std::string>::const_iterator end = _path.end();

   std::string currentRep("CurrentRepertoire");
   const RepertoireDataItem* crdi = dynamic_cast<const RepertoireDataItem*>(
      c->symTable.getEntry(currentRep));
   if (crdi == 0) {
//      std::string mes = 
//	 "dynamic cast of DataItem to RepertoireDataItem failed";
      std::string mes = 
	 "dynamic cast of DataItem to RepertoireDataItem failed: the current LensContext object does not have the repertoire with name " + (currentRep);
      throwError(mes);
   }
   Repertoire *current = crdi->getRepertoire();
   bool found=false;

   // if first entry == ".", then skip since current is correct
   i = begin;
   if (*i == ".") {
      ++begin;
      found = true;
   }
   // traverse path in the current Rep
   findRep(found, begin, end, &current);
   // If not found, search in the symbol table
   if (!found) {
      const DataItem* di = c->symTable.getCurrentScopeEntry(*begin);
      if (di) {
	 // It is in the symbol table
	 crdi = dynamic_cast<const RepertoireDataItem*>(di);
	 if (crdi == 0) {
	    std::string mes = 
	       "dynamic cast of DataItem to RepertoireDataItem failed";
	    throwError(mes);
	 } else {
	    current = crdi->getRepertoire();
	    ++begin;
	    found = true;
	    findRep(found, begin, end, &current);
	 }
      }
   }
   if (found) {
      _repertoire = current;
   } else {
      std::string mes = "unable to find " + *i;
      throwError(mes);
   } 
}

void C_repname::findRep(
   bool& found, std::list<std::string>::const_iterator& begin,
   std::list<std::string>::const_iterator& end, Repertoire **current)
{ 
   for (std::list<std::string>::const_iterator i = begin; i != end; ++i) {
      // find repertoire matching current name
      found = false;
      std::list<Repertoire*> const & subs = (*current)->getSubRepertoires();
      std::list<Repertoire*>::const_iterator j, subsBegin = subs.begin(), 
	 subsEnd = subs.end();
      for(j = subsBegin; j != subsEnd; ++j) {
         if (*i == (*j)->getName()) {
            *current = *j;
            found = true;
            break;
         }
      }
      if (!found) {
         break;
      }
   }
}

C_repname::C_repname(const C_repname& rv)
   : C_production(rv), _name(0), _preamble(0), _repertoire(rv._repertoire)
{
   if (rv._preamble) {
      _preamble = rv._preamble->duplicate();
   }
   if (rv._name) {
      _name = new std::string(*(rv._name));
   }
}


C_repname::C_repname(C_preamble* pr, std::string* name, SyntaxError* error)
   : C_production(error), _name(name), _preamble(pr), _repertoire(0)
{
}


C_repname::C_repname(std::string *name, SyntaxError * error)
   : C_production(error), _name(name), _preamble(0), _repertoire(0)
{
}


C_repname::C_repname(SyntaxError * error)
   : C_production(error), _name(0), _preamble(0), _repertoire(0)
{
   _name = new std::string(".");
}


C_repname* C_repname::duplicate() const
{
   return new C_repname(*this);
}


C_repname::~C_repname()
{
   delete _preamble;
   delete _name;
}

void C_repname::checkChildren() 
{
   if (_preamble) {
      _preamble->checkChildren();
      if (_preamble->isError()) {
         setError();
      }
   }
} 

void C_repname::recursivePrint() 
{
   if (_preamble) {
      _preamble->recursivePrint();
   }
   printErrorMessage();
} 
