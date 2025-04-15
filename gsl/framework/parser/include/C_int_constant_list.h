// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_int_constant_list_H
#define C_int_constant_list_H
#include "Copyright.h"

#include <list>
#include "C_production_adi.h"

class LensContext;
class ArrayDataItem;
class SyntaxError;

class C_int_constant_list : public C_production_adi
{
   public:
      C_int_constant_list(const C_int_constant_list&);
      C_int_constant_list(int, SyntaxError *);
      C_int_constant_list(C_int_constant_list*, int, SyntaxError *);
      virtual C_int_constant_list* duplicate() const;
      std::list<int>* releaseList();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual void internalExecute(LensContext *, ArrayDataItem *);
      virtual ~C_int_constant_list();
      std::list<int> * getList() const {
	 return _list;
      }
      int getOffset() {
	 return _offset;
      }

   private:
      std::list<int>* _list;
      int _offset;
};
#endif
