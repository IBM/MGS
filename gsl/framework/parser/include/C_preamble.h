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

#ifndef C_preamble_H
#define C_preamble_H
#include "Copyright.h"

#include <string>
#include <list>
#include "C_production.h"

class LensContext;
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
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::list<std::string>* getListStrings() const;

   private:
      std::list<std::string>* _listStrings;
};
#endif
