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

#ifndef UserFunctionCall_H
#define UserFunctionCall_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "Class.h"

class UserFunctionCall {

   public:
      UserFunctionCall(const std::string& name);
      virtual void duplicate(std::unique_ptr<UserFunctionCall>&& rv) const;
      virtual ~UserFunctionCall();
     
      std::string getName() const {
	 return _name;
      }

   protected:
      std::string _name;
};


#endif // UserFunctionCall _H
