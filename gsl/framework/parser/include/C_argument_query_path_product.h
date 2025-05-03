// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_ARGUMENT_QUERY_PATH_PRODUCT_H
#define C_ARGUMENT_QUERY_PATH_PRODUCT_H
#include "Copyright.h"

#include <string>
#include "C_argument.h"

class C_query_path_product;
class Publisher;
class DataItem;
class PublisherDataItem;
class C_argument;
class GslContext;
class ServiceDataItem;
class TriggerTypeDataItem;
class SyntaxError;

class C_argument_query_path_product: public C_argument
{
   public:
      C_argument_query_path_product(const C_argument_query_path_product&);
      C_argument_query_path_product(C_query_path_product *, SyntaxError *);
      virtual ~C_argument_query_path_product();
      virtual C_argument_query_path_product* duplicate() const;
      virtual void internalExecute(GslContext *);
      virtual void checkChildren();
      virtual void recursivePrint();

      // accessors
      C_query_path_product *getQueryPathProduct() { 
	 return _queryPathProduct;
      }
      DataItem* getArgumentDataItem() const;

   private:
      C_query_path_product* _queryPathProduct;
      ServiceDataItem* _serviceDI;
      TriggerTypeDataItem* _triggerDI;
};
#endif
