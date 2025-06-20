// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ServiceAcceptor_H
#define ServiceAcceptor_H
#include "Copyright.h"

#include <string>
class Service;

class ServiceAcceptor
{

   public:
      virtual void acceptService(Service* service, 
				 const std::string& name) = 0;
      virtual ~ServiceAcceptor() {}
};
#endif
