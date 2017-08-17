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

#ifndef ParameterSetType_H
#define ParameterSetType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class ParameterSetType : public LensType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~ParameterSetType();        

      virtual std::string getDescriptor() const;

      // This method returns if the pointer of the specific dataType 
      // is meant to be owned by the class.
      virtual bool shouldBeOwned() const;
};

#endif // ParameterSetType_H
