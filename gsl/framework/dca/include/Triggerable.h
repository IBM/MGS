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
	 std::auto_ptr<NDPairList>& ndpList) = 0;
      virtual ~Triggerable() {};
};
#endif
