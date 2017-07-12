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

#ifndef ConnectionException_H
#define ConnectionException_H
#include "Mdl.h"

#include "GeneralException.h"

class ConnectionException : public GeneralException {
   public:
      ConnectionException(const std::string& error);
      ~ConnectionException();        
};

#endif // ConnectionException_H
