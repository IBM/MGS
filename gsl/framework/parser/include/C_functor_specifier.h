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

#ifndef C_functor_specifier_H
#define C_functor_specifier_H
#include "Copyright.h"

#include <memory>
#include "C_production.h"

class C_declarator;
class C_argument_list;
class LensContext;
class DataItem;
class SyntaxError;

class C_functor_specifier : public C_production
{
   public:
      C_functor_specifier(const C_functor_specifier&);
      C_functor_specifier(C_declarator *, C_argument_list *, SyntaxError *);
      virtual ~C_functor_specifier();
      virtual C_functor_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // Will return null if void
      const DataItem* getRVal() const;

   private:
      C_declarator* _functorDeclarator;
      C_argument_list* _argumentList;
      std::auto_ptr<DataItem> _rval;
};
#endif
