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

#ifndef C_DEFINITION_FUNCTOR_H
#define C_DEFINITION_FUNCTOR_H
#include "Copyright.h"

#include <memory>
#include <map>
#include "C_definition.h"

class C_functor_definition;
class LensContext;
class SyntaxError;

class C_definition_functor : public C_definition
{
   public:
      C_definition_functor(const C_definition_functor&);
      C_definition_functor(C_functor_definition *, SyntaxError *);
      virtual C_definition_functor* duplicate() const;
      virtual ~C_definition_functor();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

   private:
      C_functor_definition* _functor_def;
};
#endif
