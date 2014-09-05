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

#ifndef UserFunctionCall_H
#define UserFunctionCall_H
#include "Mdl.h"

#include <memory>
#include <string>
#include "Class.h"

class UserFunctionCall {

   public:
      UserFunctionCall(const std::string& name);
      virtual void duplicate(std::auto_ptr<UserFunctionCall>& rv) const;
      virtual ~UserFunctionCall();
     
      std::string getName() const {
	 return _name;
      }

   protected:
      std::string _name;
};


#endif // UserFunctionCall _H
