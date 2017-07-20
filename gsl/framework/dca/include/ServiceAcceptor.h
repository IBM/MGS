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
