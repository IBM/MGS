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

#ifndef USERINTERFACE_H
#define USERINTERFACE_H
#include "Copyright.h"

#include <string>

class Simulation;

class UserInterface
{
   public:
      virtual void getUserInput(Simulation& sim)=0;
      virtual std::string getCommand()=0;
      virtual ~UserInterface() {}
};
#endif
