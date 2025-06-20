// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_H
#define C_ARGUMENT_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_production.h"

class GslContext;
class DataItem;

class C_argument : public C_production
{
   public:
	   //TUAN: these items are not used so far
	   //_NVPAIRLIST, _NVPAIR 
      enum Type
      {
         _CONSTANT,
         _DECLARATOR,
         _DECL_ARGS,
         _GRIDSET,
         _NODESET,
         _REL_NODESET,
         _STRING,
         _MATRIX,
         _PSET,
         _FUNCTOR,
         _ARG_LIST,
         _NVPAIR,
         _NVPAIRLIST,
         _NDPAIR,
         _NDPAIRLIST,
         _NULL,
         _EDGESET,
         _QUERY_PATH_PRODUCT,
         _GRANULE_MAPPER
      };

      C_argument(Type, SyntaxError* error);
      C_argument(const C_argument&);
      virtual ~C_argument();
      virtual C_argument* duplicate() const = 0;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      Type getType() { 
	 return _type;
      }
      virtual DataItem* getArgumentDataItem() const = 0;

   protected:
      Type _type;
};
#endif
