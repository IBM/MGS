// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_LIST_H
#define C_ARGUMENT_LIST_H
#include "Copyright.h"

#include <list>
#include <vector>

#include "C_production.h"

class C_argument;
class LensContext;
class DataItem;

class C_argument_list : public C_production
{
   public:
      C_argument_list(const C_argument_list&);
      C_argument_list(C_argument *, SyntaxError *);
      C_argument_list(C_argument_list *, C_argument *, SyntaxError *);
      std::list<C_argument*>* releaseList();
      virtual ~C_argument_list();
      virtual C_argument_list* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      std::vector<DataItem*>* getVectorDataItem() { 
	 return &_dataitem_list; 
      }

   private:
      std::list<C_argument*>* _list;
      std::vector<DataItem*> _dataitem_list;
};
#endif
