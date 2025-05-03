// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_repname_H
#define C_repname_H
#include "Copyright.h"

#include <string>
#include <list>
#include "C_production.h"

class C_preamble;
class GslContext;
class Repertoire;
class SyntaxError;

class C_repname : public C_production
{
   public:
      C_repname(const C_repname&);
      C_repname(C_preamble *, std::string *, SyntaxError *);
      C_repname(std::string *, SyntaxError *);
      C_repname(SyntaxError *);
      virtual ~C_repname ();
      virtual C_repname* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<std::string>& getPath() {
	 return _path;
      }
      Repertoire *getRepertoire() { 
	 return _repertoire;
      }
      void findRep(bool&, std::list<std::string>::const_iterator&, 
		   std::list<std::string>::const_iterator& , Repertoire**);

   private:
      std::string* _name;
      C_preamble* _preamble;
      std::list<std::string> _path;
      Repertoire* _repertoire;
};
#endif
