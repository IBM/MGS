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

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "SymbolTable.h"
#include "DataItem.h"
#include "SyntaxErrorException.h"
#include <iostream>
#include <sstream>

std::ostream &operator<<(std::ostream &ost, SymbolTable &st)
{
   return st.print(ost);
}


SymbolTable::SymbolTable()
: _compTrigNum(0)
{
   _scopes.push_back(new Scope);
}


std::ostream &SymbolTable::print(std::ostream & ost)
{
   int scopeLevel = 1;
   std::list<Scope*>::reverse_iterator scopeEnd=_scopes.rend();
   for (std::list<Scope*>::reverse_iterator scope=_scopes.rbegin();scope!=scopeEnd ; ++scope) {
      ost << "Scope level: "<<scopeLevel++<<" of "<<_scopes.size()<<std::endl;
      Scope::iterator end=(*scope)->end();
      for(Scope::iterator entry=(*scope)->begin(); entry!=end;++entry) {
         ost << (*entry).first <<" = " << *(*entry).second<<std::endl;
      }
   }
   return ost;

}

void SymbolTable::printOut()
{
   int scopeLevel = 1;
   std::list<Scope*>::reverse_iterator scopeEnd=_scopes.rend();
   for (std::list<Scope*>::reverse_iterator scope=_scopes.rbegin();scope!=scopeEnd ; ++scope) {
      std::cout << "Scope level: "<<scopeLevel++<<" of "<<_scopes.size()<<std::endl;
      Scope::iterator end=(*scope)->end();
      for(Scope::iterator entry=(*scope)->begin(); entry!=end;++entry) {
         std::cout << (*entry).first <<" = " << *(*entry).second<<std::endl;
      }
   }

}


SymbolTable::SymbolTable(const SymbolTable& st)
   : _compTrigNum(st._compTrigNum)
{
   std::list<Scope*>::const_iterator scope, scopeEnd = st._scopes.end();
   for (scope = st._scopes.begin(); scope != scopeEnd; ++scope) {
      Scope* newScope = new Scope;
      _scopes.push_back(newScope);

      Scope::const_iterator it, end = (*scope)->end();
      for(it = (*scope)->begin(); it != end; ++it) {
         std::unique_ptr<DataItem> apdi;
         const Scope::value_type& vt = (*it);
         vt.second->duplicate(apdi);
         newScope->insert(Scope::value_type(vt.first, apdi.release()));
      }
   }

}


SymbolTable::~SymbolTable()
{
   std::list<Scope*>::iterator scopeIt, scopeEnd=_scopes.end();
   for (scopeIt = _scopes.begin(); scopeIt != scopeEnd; ++scopeIt) {
      Scope::iterator it, end = (*scopeIt)->end();
      for(it = (*scopeIt)->begin(); it != end; ++it) {
	 delete it->second;
      }
      delete (*scopeIt);
   }
}


DataItem* SymbolTable::getEntry(const std::string& symbol)
{
   DataItem *di = 0;
   Scope::iterator de;
   std::list<Scope*>::reverse_iterator scope, scopeEnd = _scopes.rend();
   for (scope = _scopes.rbegin(); scope != scopeEnd; ++scope) {
      de = (*scope)->find(symbol);
      if (de != (*scope)->end()) {
         di = de->second;
         break;
      }
   }
   return di;
}

DataItem* SymbolTable::getCurrentScopeEntry(const std::string& symbol)
{
   DataItem *di = 0;
   Scope::iterator de;
   Scope* s = *(_scopes.rbegin());
   de = s->find(symbol);
   if (de != s->end()) {
      di = de->second;
   }
   return di;
}

void SymbolTable::addEntry(const std::string& symbol, 
			   std::unique_ptr<DataItem> &value)
{
   Scope* scope = _scopes.back();

   // look for symbol in current scope
   Scope::iterator de = scope->find(symbol);
   if (de != scope->end() ) {
      throw SyntaxErrorException("attempt to add duplicate symbol: " + 
				 symbol);
   } else {
      scope->insert(Scope::value_type(symbol, value.release()));
   }
}

void SymbolTable::updateEntry(const std::string& symbol, 
			      std::unique_ptr<DataItem>& value)
{
   bool found = false;
   Scope::iterator de;
   std::list<Scope*>::reverse_iterator scope, scopeEnd = _scopes.rend();
   for (scope = _scopes.rbegin(); scope != scopeEnd; ++scope) {
      de = (*scope)->find(symbol);
      if (de != (*scope)->end()) {
	 delete de->second;
	 de->second = value.release();
	 found = true;
         break;
      }
   }

   if (!found) {
      throw SyntaxErrorException("attempt to update a nonexistant symbol: " + 
				 symbol);
   }
}

int SymbolTable::addCompositeTrigger(std::unique_ptr<DataItem> &value)
{
   int retval = 0;
   Scope* scope = _scopes.back();
   
   std::ostringstream os;
   os << "_autoGenCompTrig" << _compTrigNum++;
   std::string symbol = os.str();
   // look for symbol in current scope
   Scope::iterator de = scope->find(symbol);
   if (de != scope->end() ) {
      // Oops! There is already an entry!
      std::cout << "Attempt to add duplicate symbol: " << symbol << std::endl;
      retval = 1;
   } else {
      scope->insert(Scope::value_type(symbol, value.release()));
   }
   return retval;
}


void SymbolTable::addLocalScope()
{
   _scopes.push_back(new Scope);
}


void SymbolTable::removeLocalScope()
{
   Scope* scope = _scopes.back();
   Scope::iterator it, end = scope->end();
   for (it = scope->begin(); it != end; ++it) {
      delete it->second;
   }
   _scopes.pop_back();
   delete scope;
}
