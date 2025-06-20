// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_index_set_specifier_H
#define C_index_set_specifier_H
#include "Copyright.h"

#include <vector>
#include "C_production.h"

class C_index_set;
class GslContext;
class SyntaxError;

class C_index_set_specifier : public C_production
{
   public:
      C_index_set_specifier(const C_index_set_specifier&);
      C_index_set_specifier(C_index_set *, SyntaxError *);
      virtual ~C_index_set_specifier();
      virtual C_index_set_specifier* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::vector<int>& getIndices() const;

   private:
      C_index_set* _indexSet;
      std::vector<int> _indices;
};
#endif
