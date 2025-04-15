// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_argument_query_path_product.h"
#include "C_query_path_product.h"
#include "Service.h"
#include "TriggerType.h"
#include "ServiceDataItem.h"
#include "TriggerTypeDataItem.h"
#include "SyntaxError.h"

void C_argument_query_path_product::internalExecute(LensContext *c)
{
   _queryPathProduct->execute(c);
   if (_queryPathProduct->getType() == C_query_path_product::_SERVICE) {
      _serviceDI = new ServiceDataItem;
      _serviceDI->setService(_queryPathProduct->getService());
   }
   else if (_queryPathProduct->getType() == C_query_path_product::_TRIGGER) {
      _triggerDI = new TriggerTypeDataItem;
      _triggerDI->setTriggerType(_queryPathProduct->getTriggerDescriptor());
   }
}


C_argument_query_path_product::C_argument_query_path_product(
   const C_argument_query_path_product& rv)
   : C_argument(rv), _queryPathProduct(0), _serviceDI(0), _triggerDI(0)
{
   if (rv._queryPathProduct) {
      _queryPathProduct = rv._queryPathProduct->duplicate();
   }
   if (rv._serviceDI) {
      std::unique_ptr<DataItem> cc_di;
      rv._serviceDI->duplicate(cc_di);
      _serviceDI = dynamic_cast<ServiceDataItem*>(cc_di.release());
   }
   if (rv._triggerDI) {
      std::unique_ptr<DataItem> cc_di;
      rv._triggerDI->duplicate(cc_di);
      _triggerDI = dynamic_cast<TriggerTypeDataItem*>(cc_di.release());
   }
}


C_argument_query_path_product::C_argument_query_path_product(
   C_query_path_product *qpp,  SyntaxError * error)
   : C_argument(_QUERY_PATH_PRODUCT, error), _queryPathProduct(qpp), 
     _serviceDI(0), _triggerDI(0)
{
}


C_argument_query_path_product* C_argument_query_path_product::duplicate() const
{
   return new C_argument_query_path_product(*this);
}


C_argument_query_path_product::~C_argument_query_path_product()
{
   delete _queryPathProduct;
   delete _serviceDI;
   delete _triggerDI;
}


DataItem* C_argument_query_path_product::getArgumentDataItem() const
{
   DataItem* rval = 0;
   if (_serviceDI) rval = _serviceDI;
   else if (_triggerDI) rval = _triggerDI;
   return rval;
}

void C_argument_query_path_product::checkChildren() 
{
   if (_queryPathProduct) {
      _queryPathProduct->checkChildren();
      if (_queryPathProduct->isError()) {
         setError();
      }
   }
} 

void C_argument_query_path_product::recursivePrint() 
{
   if (_queryPathProduct) {
      _queryPathProduct->recursivePrint();
   }
   printErrorMessage();
} 
