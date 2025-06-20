// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
