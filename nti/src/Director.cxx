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

#include "Director.h"
#include "Communicator.h"
#include "Sender.h"
#include "Receiver.h"
#include <cassert>

Director::Director(Communicator* communicator) :
  _communicator(communicator), _done(false)
{
}

Director::~Director()
{
  clearCommunicationCouples();
}

void Director::addCommunicationCouple(Sender* sender, Receiver* receiver)
{
  CommunicationCouple* couple = new CommunicationCouple(sender, receiver);
  _couples.push_back(couple);
}

void Director::clearCommunicationCouples()
{
  std::vector<CommunicationCouple*>::iterator iter = _couples.begin(),
    end = _couples.end();
  for (; iter!=end; ++iter) delete (*iter);  
  _couples.clear();
}

void Director::iterate()
{
  std::vector<CommunicationCouple*>::iterator iter = _couples.begin(),
                                                 end = _couples.end();
  CommunicatorFunction cf;
  CommunicatorFunction cf2;

  int i=0;

  for (; iter!=end; ++iter) {
    Sender* sender = (*iter)->getSender();
    Receiver* receiver = (*iter)->getReceiver();
    int iend = sender->getNumberOfSendCycles();
    assert (iend == receiver->getNumberOfReceiveCycles() );
    for (int i=0; i<iend; ++i) {
      int jend = sender->getNumberOfSendPhasesPerCycle(i);
      assert (jend == receiver->getNumberOfReceivePhasesPerCycle(i) );
      
      for (int j=0; j != jend; ++j) {
	sender->prepareToSend(i, j, cf);
	receiver->prepareToReceive(i, j, cf2);
	assert(cf==cf2);
	(_communicator->*cf)(sender, receiver, i, j);
	receiver->finalizeReceive(i, j);
	MPI_Barrier(MPI_COMM_WORLD);
      }
    }
  }
}
