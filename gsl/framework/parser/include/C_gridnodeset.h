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

#ifndef C_gridnodeset_H
#define C_gridnodeset_H
#include "Copyright.h"

#include <vector>
#include "C_production.h"

class C_index_set;
class LensContext;
class SyntaxError;

class C_gridnodeset : public C_production
{
   public:
      C_gridnodeset(const C_gridnodeset&);
      C_gridnodeset(C_index_set *, SyntaxError *);
      virtual ~C_gridnodeset();
      virtual C_gridnodeset* duplicate() const;
      virtual void internalExecute(LensContext *);
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
