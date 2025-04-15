// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
