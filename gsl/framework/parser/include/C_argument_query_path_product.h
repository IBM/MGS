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
class LensContext;
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
      virtual void internalExecute(LensContext *);
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
