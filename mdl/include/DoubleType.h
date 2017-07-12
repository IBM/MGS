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

#ifndef DoubleType_H
#define DoubleType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "DataType.h"

class DoubleType : public DataType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual bool isBasic() const;
      virtual ~DoubleType();        

      virtual std::string getDescriptor() const;
      virtual std::string getArrayDataItemString() const;
      virtual std::string getCapitalDescriptor() const;
      virtual std::string getDataItemString() const;
      virtual std::string getDataItemFunctionString() const;
      virtual std::string getArrayInitializerString(
	 const std::string& name,
	 const std::string& arrayName,
	 int level) const;
      virtual std::string getInitializationDataItemString() const;
};

#endif // DoubleType_H
