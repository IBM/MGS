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

#ifndef C_SERVICE_H
#define C_SERVICE_H
#include "Copyright.h"

#include "C_production.h"
#include <string>

class C_query_path_product;
class C_argument_declarator;
class C_argument_string;
class C_declarator;
class LensContext;
class DataItem;
class Service;
class SyntaxError;

class C_service : public C_production
{
   public:
      C_service(const C_service&);
      C_service(C_query_path_product*, SyntaxError *);
      C_service(C_declarator*, std::string*, SyntaxError *);
      virtual ~C_service();
      virtual C_service* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      Service* getService() { 
	 return _service; 
      }

   private:
      C_query_path_product* _queryPathProduct;
      C_declarator* _declarator;
      std::string* _string;
      Service* _service;
};
#endif
