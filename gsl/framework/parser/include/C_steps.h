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

#ifndef C_steps_H
#define C_steps_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_int_constant_list;
class LensContext;
class SyntaxError;

class C_steps : public C_production
{
   public:
      C_steps(const C_steps&);
      C_steps(C_int_constant_list *, SyntaxError *);
      virtual ~C_steps();
      virtual C_steps* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::list<int>* getListInt() const;

   private:
      C_int_constant_list* _cIntConstList;
};
#endif
