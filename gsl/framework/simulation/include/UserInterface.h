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
