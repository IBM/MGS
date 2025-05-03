// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_preamble_H
#define C_preamble_H
#include "Copyright.h"

#include <string>
#include <list>
#include "C_production.h"

class GslContext;
class SyntaxError;

class C_preamble : public C_production
{
   public:
      C_preamble(const C_preamble&);
      C_preamble(std::string *, SyntaxError *);
      C_preamble(SyntaxError *);
      C_preamble(C_preamble *, std::string *, SyntaxError *);
      std::list<std::string>* releaseList();
      virtual ~C_preamble();
      virtual C_preamble* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<std::string>* getListStrings() const;

   private:
      std::list<std::string>* _listStrings;
};
#endif
