// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef DIRECTOR_H
#define DIRECTOR_H

#include <mpi.h>
#include <vector>
#include <map>

#include "CommunicationCouple.h"

class Communicator;
class Sender;
class Receiver;

class Director
{
   public:
    Director(Communicator* communicator);
    ~Director();      
    void addCommunicationCouple(Sender* sender, Receiver* receiver);
    void clearCommunicationCouples();
    void iterate();
     
   protected:
    Communicator* _communicator;
    std::vector<CommunicationCouple*> _couples;
    
    bool _done;
};

#endif

