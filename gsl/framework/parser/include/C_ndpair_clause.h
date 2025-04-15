// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef _C_NDPAIR_CLAUSE_H_
#define _C_NDPAIR_CLAUSE_H_
#include "Copyright.h"

#include "C_production.h"
#include <memory>

class C_name;
class C_argument;
class NDPair;
class LensContext;
class SyntaxError;

class C_ndpair_clause : public C_production
{
   public:
      C_ndpair_clause(const C_ndpair_clause&);
      C_ndpair_clause(C_name *, C_argument *, SyntaxError *);
      virtual ~C_ndpair_clause();
      virtual C_ndpair_clause* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const NDPair& getNDPair() {
	 return *_ndpair;
      }
      void releaseNDPair(std::unique_ptr<NDPair>& ndp);

   private:
      NDPair* _ndpair;
      C_name* _name;
      C_argument* _argument;
};
#endif
