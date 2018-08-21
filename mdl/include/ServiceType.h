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

#ifndef ServiceType_H
#define ServiceType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class ServiceType : public LensType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~ServiceType();        

      virtual std::string getDescriptor() const;
};

#endif // ServiceType_H
