// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_constant_list_H
#define C_constant_list_H
#include "Copyright.h"

#include <list>
#include "C_production.h"

class C_constant;
class LensContext;
class SyntaxError;

class C_constant_list : public C_production
{
   public:
      C_constant_list(const C_constant_list&);
      C_constant_list(C_constant *, SyntaxError *);
      C_constant_list(C_constant_list *, C_constant *, SyntaxError *);
      virtual C_constant_list* duplicate() const;
      std::list<C_constant>* releaseList();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual ~C_constant_list();
      std::list<C_constant>* getList()const;

   private:
      std::list<C_constant>* _list;
};
#endif
