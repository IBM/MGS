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

#ifndef CharType_H
#define CharType_H
#include "Mdl.h"

#include <string>
#include <memory>
#include "SignedType.h"

class CharType : public SignedType {
   public:
      virtual void duplicate(std::auto_ptr<DataType>& rv) const;
      virtual ~CharType();        

      virtual std::string getDescriptor() const;
      virtual std::string getCapitalDescriptor() const;
};

#endif // CharType_H
