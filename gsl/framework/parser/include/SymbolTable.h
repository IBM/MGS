// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _SYMBOLTABLE_H
#define _SYMBOLTABLE_H
#include "Copyright.h"

#include <memory>
#include <string>
#include <map>
#include <list>
#include <iostream>

class DataItem;
class SymbolTable;

std::ostream &operator<<(std::ostream &, SymbolTable &);

class SymbolTable
{
   public:
      typedef std::map<std::string,DataItem*> Scope;

      SymbolTable();
      SymbolTable(const SymbolTable& st);
      ~SymbolTable();

      DataItem* getEntry(const std::string& symbol);
      DataItem* getCurrentScopeEntry(const std::string& symbol);

      void addEntry(const std::string& symbol, 
		    std::unique_ptr<DataItem>& value);
      void updateEntry(const std::string& symbol, 
		       std::unique_ptr<DataItem>& value);
      int addCompositeTrigger(std::unique_ptr<DataItem>& value);

      void addLocalScope();
      void removeLocalScope();

      std::ostream & print(std::ostream &);
      void printOut();

   private:

      std::list<Scope*> _scopes;
      // This is slightly different than what we want
      // There should be one of these for each scope.
      // But since we just want a unique number, that 
      // hasn't been used before, it is ok.
      int _compTrigNum;
};
#endif
