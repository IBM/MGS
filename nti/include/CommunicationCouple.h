// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
