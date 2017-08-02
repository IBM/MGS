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

#ifndef TriggeredFunctionShared_H
#define TriggeredFunctionShared_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "TriggeredFunction.h"

class Generatable;

class TriggeredFunctionShared : public TriggeredFunction {

   public:
      TriggeredFunctionShared(const std::string& name, RunType runType);
      virtual void duplicate(
	 std::auto_ptr<TriggeredFunction>& rv) const;
      virtual ~TriggeredFunctionShared();
      
   protected:
      virtual std::string getTriggerableType (
	 const std::string& modelName) const;     
      virtual std::string getTab() const {
	 return "\t\t";
      }

};

#endif // TriggeredFunctionShared_H
