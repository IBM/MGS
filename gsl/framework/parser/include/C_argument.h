// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_ARGUMENT_H
#define C_ARGUMENT_H
#include "Copyright.h"

#include <string>
#include <memory>
#include "C_production.h"

class LensContext;
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
      virtual void internalExecute(LensContext *);
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
