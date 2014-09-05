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

#include "C_query_path_product.h"
#include "C_query_path.h"
#include "C_declarator.h"
#include "Service.h"
#include "ServiceDescriptor.h"
#include "TriggerType.h"
#include "Publisher.h"
#include "LensContext.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "C_production.h"

#include <assert.h>

void C_query_path_product::internalExecute(LensContext *c)
{
   _queryPath->execute(c);
   _declarator->execute(c);
   std::string name = _declarator->getName();
   Publisher* p = _queryPath->getPublisher();
   const std::vector<ServiceDescriptor>& serviceDescriptors = 
      p->getServiceDescriptors();
   std::vector<ServiceDescriptor>::const_iterator iter, 
      end = serviceDescriptors.end();

   for (iter = serviceDescriptors.begin(); iter != end; ++iter) {
      if (iter->getName() == name) {
         _service = p->getService(name);
         _type = _SERVICE;
         break;
      }
   }
   if (!_service) {
      const std::vector<TriggerType*>& triggers = p->getTriggerDescriptors();
      std::vector<TriggerType*>::const_iterator iter, end = triggers.end();
      for (iter = triggers.begin(); iter != end; ++iter) {
         if ((*iter)->getName() == name) {
            _triggerType = (*iter);
            _type = _TRIGGER;
            break;
         }
      }
   }
   if (!_service && !_triggerType) {
      std::string mes = "specified product " + name + 
	 " is not available from specified publisher " + p->getName();
      throwError(mes);
   }
}


C_query_path_product::C_query_path_product(
   C_query_path* qp, C_declarator* d, SyntaxError * error)
   : C_production(error), _queryPath(qp), _declarator(d), _service(0), 
     _triggerType(0)
{
}


C_query_path_product::C_query_path_product(const C_query_path_product& rv)
   : C_production(rv), _queryPath(0), _declarator(0), _service(rv._service), 
     _triggerType(rv._triggerType)
{
   if (rv._queryPath) {
      _queryPath = rv._queryPath->duplicate();
   }
   if (rv._declarator) {
      _declarator = rv._declarator->duplicate();
   }
}


C_query_path_product* C_query_path_product::duplicate() const
{
   return new C_query_path_product(*this);
}


Service* C_query_path_product::getService()
{
   if (!_service) {
      std::string mes = "specified service " + _declarator->getName() + 
	 " is not available from specified publisher " + 
	 _queryPath->getPublisher()->getName();
      throwError(mes);
   }
   return _service;
}


TriggerType* C_query_path_product::getTriggerDescriptor()
{
   if (!_triggerType) {
      std::string mes = "specified trigger type " + _declarator->getName() + 
	 " not available from specified publisher " + 
	 _queryPath->getPublisher()->getName();
      throwError(mes);
   }
   return _triggerType;
}


C_query_path_product::~C_query_path_product()
{
   delete _queryPath;
   delete _declarator;
}

void C_query_path_product::checkChildren() 
{
   if (_queryPath) {
      _queryPath->checkChildren();
      if (_queryPath->isError()) {
         setError();
      }
   }
   if (_declarator) {
      _declarator->checkChildren();
      if (_declarator->isError()) {
         setError();
      }
   }
} 

void C_query_path_product::recursivePrint() 
{
   if (_queryPath) {
      _queryPath->recursivePrint();
   }
   if (_declarator) {
      _declarator->recursivePrint();
   }
   printErrorMessage();
} 
