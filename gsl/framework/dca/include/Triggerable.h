// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TRIGGERABLE_H
#define TRIGGERABLE_H
#include "Copyright.h"

#include <string>
#include <memory>

class Trigger;
class NDPairList;

class Triggerable
{
   public:
      virtual void addTrigger(
	 Trigger* trigger, const std::string& functionName, 
	 std::unique_ptr<NDPairList>& ndpList) = 0;
      virtual ~Triggerable() {};
};
#endif
