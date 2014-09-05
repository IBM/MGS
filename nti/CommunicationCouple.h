// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. and EPFL 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// ================================================================

#ifndef COMMUNICATIONCOUPLE_H
#define COMMUNICATIONCOUPLE_H

#include <mpi.h>

class Sender;
class Receiver;

class CommunicationCouple 
{
  public:
    CommunicationCouple(Sender* sender, Receiver* receiver) : _sender(sender), _receiver(receiver) {}
    CommunicationCouple(CommunicationCouple& cc) : _sender(cc._sender), _receiver(cc._receiver) {}
    Sender* getSender() {return _sender;}
    Receiver* getReceiver() {return _receiver;}
    ~CommunicationCouple() {}

  private: 
    Sender* _sender;
    Receiver* _receiver;
};

#endif
