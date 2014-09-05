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
      void releaseNDPair(std::auto_ptr<NDPair>& ndp);

   private:
      NDPair* _ndpair;
      C_name* _name;
      C_argument* _argument;
};
#endif
