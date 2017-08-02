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

#ifndef StringType_H
#define StringType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include <vector>
#include "DataType.h"

class StringType : public DataType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~StringType();        
      
      virtual std::string getDescriptor() const;
      virtual std::string getHeaderString(
	 std::vector<std::string>& arrayTypeVec) const;
   protected:
      virtual std::string getDataItemFunctionString() const;    
};

#endif // StringType_H
