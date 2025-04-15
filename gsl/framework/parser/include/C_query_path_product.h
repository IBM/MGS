// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_QUERY_PATH_PRODUCT_H
#define C_QUERY_PATH_PRODUCT_H
#include "Copyright.h"

#include <string>
#include <vector>
#include "C_production.h"

class C_query_path;
class C_declarator;
class LensContext;
class Service;
class TriggerType;
class SyntaxError;

class C_query_path_product : public C_production
{
   public:
      enum Type{_SERVICE, _TRIGGER};
      C_query_path_product(const C_query_path_product&);
      C_query_path_product(C_query_path*, C_declarator*, SyntaxError *);
      virtual ~C_query_path_product();
      virtual C_query_path_product* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Service* getService();
      TriggerType* getTriggerDescriptor();
      Type getType() {
	 return _type;
      }

   private:
      C_query_path* _queryPath;
      C_declarator* _declarator;
      Service* _service;
      TriggerType* _triggerType;
      Type _type;
};
#endif
