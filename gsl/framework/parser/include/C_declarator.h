// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_declarator_H
#define C_declarator_H
#include "Copyright.h"

#include <string>

#include "C_production.h"

class LensContext;
class SyntaxError;

class C_declarator : public C_production
{
   public:
      C_declarator(std::string *, SyntaxError *);
      C_declarator(const C_declarator&);
      virtual ~C_declarator();
      virtual C_declarator* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessor method
      const std::string& getName();

   private:
      std::string* _name;
};
#endif
