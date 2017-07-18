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

#ifndef SignedType_H
#define SignedType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "DataType.h"

class SignedType : public DataType {
   public:
      SignedType();
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      bool isSigned() const;
      void setSigned(bool sign);
      virtual bool isBasic() const;
      virtual ~SignedType();        

      virtual std::string getDescriptor() const;
      virtual std::string getDataItemString() const;
      virtual std::string getArrayDataItemString() const;
      virtual std::string getDataItemFunctionString() const;
      virtual std::string getArrayInitializerString(
	 const std::string& name,
	 const std::string& arrayName,
	 int level) const;
      virtual std::string getInitializationDataItemString() const;

   private:
      bool _signed;
};

#endif // SignedType_H
