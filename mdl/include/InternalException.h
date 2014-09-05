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

#ifndef InternalException_H
#define InternalException_H
#include "Mdl.h"

#include "GeneralException.h"

class InternalException : public GeneralException {
   public:
      InternalException(const std::string& error);
      ~InternalException();        
};

#endif // InternalException_H
