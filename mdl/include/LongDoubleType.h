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

#ifndef LongDoubleType_H
#define LongDoubleType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "DataType.h"

class LongDoubleType : public DataType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual bool isBasic() const;
      virtual ~LongDoubleType();        

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

#endif // LongDoubleType_H
