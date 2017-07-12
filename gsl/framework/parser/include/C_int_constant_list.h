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
