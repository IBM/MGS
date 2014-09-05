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

#ifndef C_functor_declarator_H
#define C_functor_declarator_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_functor_declarator : public C_production
{
   public:
      C_functor_declarator(const C_functor_declarator&);
      C_functor_declarator(const std::string&, SyntaxError *);
      virtual ~C_functor_declarator();
      virtual C_functor_declarator* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      std::string _id;
};
#endif
