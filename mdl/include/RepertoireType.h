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

#ifndef RepertoireType_H
#define RepertoireType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "LensType.h"

class RepertoireType : public LensType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~RepertoireType();        

      virtual std::string getDescriptor() const;
};

#endif // RepertoireType_H
