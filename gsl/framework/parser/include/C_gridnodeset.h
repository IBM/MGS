// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_gridnodeset_H
#define C_gridnodeset_H
#include "Copyright.h"

#include <vector>
#include "C_production.h"

class C_index_set;
class GslContext;
class SyntaxError;

class C_gridnodeset : public C_production
{
   public:
      C_gridnodeset(const C_gridnodeset&);
      C_gridnodeset(C_index_set *, SyntaxError *);
      virtual ~C_gridnodeset();
      virtual C_gridnodeset* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::vector<int>& getBeginCoords() {
	 return _begin;
      }
      const std::vector<int>& getIncrement() {
	 return _increment;
      }
      const std::vector<int>& getEndCoords() {
	 return _end;
      }

   private:
      C_index_set* _indexSet;
      std::vector<int> _begin;
      std::vector<int> _increment;
      std::vector<int> _end;
};
#endif
