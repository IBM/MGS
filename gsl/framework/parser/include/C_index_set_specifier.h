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

#ifndef C_index_set_specifier_H
#define C_index_set_specifier_H
#include "Copyright.h"

#include <vector>
#include "C_production.h"

class C_index_set;
class LensContext;
class SyntaxError;

class C_index_set_specifier : public C_production
{
   public:
      C_index_set_specifier(const C_index_set_specifier&);
      C_index_set_specifier(C_index_set *, SyntaxError *);
      virtual ~C_index_set_specifier();
      virtual C_index_set_specifier* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::vector<int>& getIndices() const;

   private:
      C_index_set* _indexSet;
      std::vector<int> _indices;
};
#endif
