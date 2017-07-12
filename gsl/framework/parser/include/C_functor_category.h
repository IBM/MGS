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

#ifndef C_functor_category_H
#define C_functor_category_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_functor_category : public C_production
{
   public:
      C_functor_category(const C_functor_category&);
      C_functor_category(std::string, SyntaxError *);
      virtual ~C_functor_category();
      virtual C_functor_category* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::string& getCategory() {
	 return _category;
      }

   private:
      std::string _category;
};
#endif
