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

