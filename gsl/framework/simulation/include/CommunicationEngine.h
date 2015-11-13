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

#ifndef CommunicationEngine_H
#define CommunicationEngine_H
#include "Copyright.h"
#ifdef HAVE_MPI

#include <map>
#include <vector>
#include "IIterator.h"
#include "ISender.h"
#include "IReceiver.h"
#include "StreamConstants.h"
#include "Simulation.h"
#include "OutputStream.h"
#include "MemPattern.h"
#define DIM 3

// WARNING! : MEMCPY_MARSHALL is experimental and results are not correct
// #define MEMCPY_MARSHALL

typedef std::map<int,IReceiver*> ReceiverMap;
typedef std::map<int,ISender*> SenderMap;
typedef std::vector<ISender*> SenderVector;

class CommunicationEngine
{
 public:
  // constructor needs senders and receivers connected to this machine node
  // These are simple vectors or lists - need not be maps
  // I will construct the maps I need

  class Args
    {
    public: 
      Args(int nprocs) : buffsize(0) {
	counts = new int[nprocs];
	displs = new int[nprocs];
	for (int i=0; i<nprocs; ++i) {
	  counts[i]=0;
	  displs[i]=0;
	}
      }
      void setBuffSize(int size) {
	buffsize=size;
      }
      ~Args() {
	delete [] counts;
	delete [] displs;
      }

      int buffsize;
      int* counts;
      int* displs;
    };

  CommunicationEngine(int MachineNodeCount, IIterator<ISender> *Senders, IIterator<IReceiver> *Receivers, Simulation *sim) :
    _nprocs(MachineNodeCount), _nsteps(0), _myrank(0), P2P_TAG(21), _receivermaps(0), _sendervectors(0), _wsndCounts(0), _wsndDispls(0), 
    _wrcvCounts(0), _wrcvDispls(0), _vsbuff(0), _vrbuff(0), _vOutputStream(0), _sim(sim)
    {
      _myrank = getRank();
      int tmp = _nprocs + _nprocs - 1;
      while (tmp >>= 1) _nsteps++;
      if (sim->P2P()) BuildLists(Senders, Receivers);
      if (sim->AllToAllV()) {
	BuildLists(Senders, Receivers);
	BuildArgs(Senders, Receivers);
      }
      if (sim->AllToAllW()) {
	_wsndCounts=new int[_nprocs];
	_wsndDispls=new int[_nprocs];
	_wrcvCounts=new int[_nprocs];
	_wrcvDispls=new int[_nprocs];
	for (int i=0; i<_nprocs; ++i) {
	  _wsndCounts[i]=0;
	  _wsndDispls[i]=0;
	  _wrcvCounts[i]=0;
	  _wrcvDispls[i]=0;
	}
	BuildDatatypes(Senders, Receivers);
      }
    }

  int getRank()
    {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      return rank;
    }

  /* * * This is what is called by the simulation engine to do communication at this phase * * */

  void Communicate() {

    if (_sim->AllToAllV()) {
      std::string phaseName = _sim->getPhaseName();      
      Args* snd = _vsndArgs[phaseName];
      Args* rcv = _vrcvArgs[phaseName];
      SenderMap::iterator send = _allsenders.end();

#ifdef MEMCPY_MARSHALL
      char* buff=_vsbuff;
      MemPattern* mpsEnd = _sndPatternMap[phaseName]._memPatternsEnd;
      for (MemPattern* mpptr=_sndPatternMap[phaseName]._memPatterns; mpptr!=mpsEnd; ++mpptr) {
	char* src = mpptr->orig;
	int *d, *dend=mpptr->origDisplsEnd;
	int *p, *pbeg=mpptr->pattern, *pend=mpptr->patternEnd;
	for (d=mpptr->origDispls; d!=dend; ++d) { 
	  src+=*d;
	  for (p=pbeg; p!=pend; ++p) {	    
	    //memcpy(buff, src+*p, *(++p));
		  std::copy(src+*p, src+*p+(*(++p)), buff);
	    buff+=*p;
	  }
	}
      }
      //assert(buff==_vsbuff+snd->displs[_nprocs-1]+snd->counts[_nprocs-1]);
#else
      _vOutputStream->reset();
      for (SenderMap::iterator s = _allsenders.begin(); s != send; ++s) {
	ISender* isndr = s->second;
	int i=isndr->getRank();
	if (snd->counts[i])
	  isndr->pack(_vOutputStream);
      }
#endif

#ifdef VERBOSE
      double now, then;
      /* MPI_W begin: measure MPI collective communication */
      now = MPI_Wtime();
#endif
      MPI_Alltoallv((void*)_vsbuff, snd->counts, snd->displs, MPI_CHAR,
		    (void*)_vrbuff, rcv->counts, rcv->displs, MPI_CHAR, _phaseCommunicators[phaseName]);

#ifdef VERBOSE
      then = MPI_Wtime();
      collectivesElapsed += (then-now);
      collectivesElapsedMap[phaseName] += (then-now);
      /* MPI_W end: measure MPI collective communication */
#endif

#ifdef MEMCPY_MARSHALL
      buff=_vrbuff;
      mpsEnd = _rcvPatternMap[phaseName]._memPatternsEnd;
      for (MemPattern* mpptr=_rcvPatternMap[phaseName]._memPatterns; mpptr!=mpsEnd; ++mpptr) {
	char* dst = mpptr->orig;       
	int *d, *dend=mpptr->origDisplsEnd;
	int *p, *pbeg=mpptr->pattern, *pend=mpptr->patternEnd;
	for (d=mpptr->origDispls; d!=dend; ++d) { 
	  dst+=*d;
	  for (p=pbeg; p!=pend; ++p) {
	    //memcpy(dst+*p, buff, *(++p));
		  std::copy(buff, buff+(*(++p)), dst+*p);
	    buff+=*p;
	  }
	}
      }
      //assert(buff==_vrbuff+rcv->displs[_nprocs-1]+rcv->counts[_nprocs-1]);
#else
      ReceiverMap::iterator rend = _allreceivers.end();
      for (ReceiverMap::iterator r = _allreceivers.begin(); r != rend; ++r) {
	IReceiver* ircvr = r->second;
	int i=ircvr->getRank();
	if (rcv->counts[i]) {
	  ircvr->reset();
	  ircvr->receive(_vrbuff+rcv->displs[i], rcv->counts[i]);
	}
      }
#endif
    }

    else if (_sim->AllToAllW()) {
      std::string phaseName = _sim->getPhaseName();
#ifdef VERBOSE
      double now, then;
      /* MPI_W begin: measure MPI collective communication */
      now = MPI_Wtime();
#endif
      MPI_Alltoallw(MPI_BOTTOM, _wsndCounts, _wsndDispls, _wsndTypes[phaseName],
		    MPI_BOTTOM, _wrcvCounts, _wrcvDispls, _wrcvTypes[phaseName], _phaseCommunicators[phaseName]);
#ifdef VERBOSE
      then = MPI_Wtime();
      collectivesElapsed += (then-now);
      /* MPI_W end: measure MPI collective communication */
#endif
    }

    else if (_sim->P2P()) {
      // need to reset ALL my receivers
      ReceiverMap::iterator rend = _allreceivers.end();
      for (ReceiverMap::iterator r = _allreceivers.begin(); r != rend; ++r) r->second->reset();
      // now go through the steps of the butterfly algorithm
      for (int i=0; i<_nsteps; ++i) {
	if (SendFirst(i)) {
	  Send(i);
	  Receive(i);
	} else {
	  Receive(i);
	  Send(i);
	}
	//MPI_Barrier(MPI_COMM_WORLD);  // use barrier outside CommunicationEngine to have loose synchronization
      }
    }
  }
    
  /* * * based on myrank - for this step, do I send first, or receive first? * * */

  bool SendFirst(int step) {
    int mask = 1 << step;
    return (_myrank & mask) == 0;
  }

  /* * * One time procedure to allocate all datatypes for AllToAllV communication * * */

  void BuildArgs(IIterator<ISender> *Senders, IIterator<IReceiver> *Receivers) {
    std::vector<std::string> const & phaseNames = _sim->getPhaseNames();
    std::vector<std::string>::const_iterator iter, end=phaseNames.end();
    Args* args = new Args(_nprocs);

    int maxSendBuffSize=1;
    int maxRecvBuffSize=1;
    for (iter=phaseNames.begin(); iter!=end; ++iter) {
      _phaseCommunicators[*iter]=MPI_Comm();
      MPI_Comm_dup(MPI_COMM_WORLD, &_phaseCommunicators[*iter]);
#ifdef VERBOSE
      collectivesElapsedMap[*iter]=0;
#endif
      int buffsize=0;
      int rank=-1;
#ifdef MEMCPY_MARSHALL
      int nSndPatterns = 0;
      int nRcvPatterns = 0;
#endif
      for (ISender *s = Senders->getFirst(); s != NULL; s = Senders->getNext()) {
	assert(rank<s->getRank()); // necessary for correct displacement calculation below
	rank=s->getRank();
#ifdef MEMCPY_MARSHALL
        nSndPatterns += s->getPatternCount(*iter);
#endif
	//std::cerr<<*iter<<"Send : ";
	buffsize+=(args->counts[rank] = s->getByteCount(*iter));
      }
      
      for (int i=1; i<_nprocs; ++i)
	args->displs[i]=args->displs[i-1]+args->counts[i-1];
      args->setBuffSize(buffsize);
      if (buffsize>maxSendBuffSize) maxSendBuffSize=buffsize;
      _vsndArgs[*iter]=args; 	
      args=new Args(_nprocs);
      //std::cerr<<(*iter)<<" sendbuff("<<rank<<"): "<<buffsize<<std::endl;
      buffsize=0;
      rank=-1;
      
      for (IReceiver *r = Receivers->getFirst(); r != NULL; r = Receivers->getNext()) {
	assert(rank<r->getRank()); // necessary for correct displacement calculation below
	rank=r->getRank();
#ifdef MEMCPY_MARSHALL
	nRcvPatterns += r->getPatternCount(*iter);
#endif
	//std::cerr<<*iter<<"Receive : ";
	buffsize+=(args->counts[rank] = r->getByteCount(*iter));
      }

      for (int i=1; i<_nprocs; ++i)
	args->displs[i]=args->displs[i-1]+args->counts[i-1];
      args->setBuffSize(buffsize);
      if (buffsize>maxRecvBuffSize) maxRecvBuffSize=buffsize;
      _vrcvArgs[*iter]=args; 	
      args=new Args(_nprocs);
      //std::cerr<<(*iter)<<" recvbuff("<<rank<<"): "<<buffsize<<std::endl;
      buffsize=0;


#ifdef MEMCPY_MARSHALL
      MemPatternPointers& smpptr = _sndPatternMap[*iter] = MemPatternPointers();
      if (nSndPatterns>0) {
	MemPattern* mp = smpptr._memPatterns = new MemPattern[nSndPatterns];
	smpptr._memPatternsEnd = smpptr._memPatterns + nSndPatterns;
	for (ISender *s = Senders->getFirst(); s != NULL; s = Senders->getNext()) {
	  mp = s->getMemPatterns(*iter, mp);
	}
	assert(mp==smpptr._memPatternsEnd);
      }

      MemPatternPointers& rmpptr =_rcvPatternMap[*iter] = MemPatternPointers();
      if (nRcvPatterns>0) {
	MemPattern* mp = rmpptr._memPatterns = new MemPattern[nRcvPatterns];
	rmpptr._memPatternsEnd = rmpptr._memPatterns + nRcvPatterns;
	for (IReceiver *r = Receivers->getFirst(); r != NULL; r = Receivers->getNext()) {
	  mp = r->getMemPatterns(*iter, mp);
	}
	assert(mp==rmpptr._memPatternsEnd);
      }
#endif
    }
    _vsbuff = new char[maxSendBuffSize];
    _vrbuff = new char[maxRecvBuffSize];
    _vOutputStream = new OutputStream(_vsbuff);
    delete args;
  }

  /* * * One time procedure to allocate all datatypes for AllToAllW communication * * */

  void BuildDatatypes(IIterator<ISender> *Senders, IIterator<IReceiver> *Receivers) {

    std::vector<std::string> const & phaseNames = _sim->getPhaseNames();
    std::vector<std::string>::const_iterator iter, end=phaseNames.end();

    for (iter=phaseNames.begin(); iter!=end; ++iter) {
      MPI_Datatype* sndTypePtr = _wsndTypes[*iter]=new MPI_Datatype[_nprocs];
      MPI_Datatype* rcvTypePtr = _wrcvTypes[*iter]=new MPI_Datatype[_nprocs];
      
      for (int i=0; i<_nprocs; ++i) {
	MPI_Type_contiguous(0, MPI_CHAR, &sndTypePtr[i]);
	MPI_Type_commit(&sndTypePtr[i]);
	MPI_Type_contiguous(0, MPI_CHAR, &rcvTypePtr[i]);
	MPI_Type_commit(&rcvTypePtr[i]);
      }
      
      for (ISender *s = Senders->getFirst(); s != NULL; s = Senders->getNext()) {
	int rank = s->getRank();
	sndTypePtr=&_wsndTypes[*iter][rank];
	assert(rank<_nprocs);
	if (s->getWSendType(*iter, sndTypePtr)) _wsndCounts[rank]=1;
      }
      for (IReceiver *r = Receivers->getFirst(); r != NULL; r = Receivers->getNext()) {
	int rank = r->getRank(); 
	rcvTypePtr=&_wrcvTypes[*iter][rank];
	assert(rank<_nprocs);
	if (r->getWReceiveType(*iter, rcvTypePtr)) _wrcvCounts[rank]=1;
      }
    }
  }

  /* * * One time procedure to allocate all lists and maps needed during each phase of butterfly * * *
   * * * This does all pre-culling so run-time is optimal                                        * * */

  void BuildLists(IIterator<ISender> *Senders, IIterator<IReceiver> *Receivers) {
    _sendervectors = new SenderVector[_nsteps];
    _receivermaps = new ReceiverMap[_nsteps];

    for (int step=0; step<_nsteps; step++) {
      int mask = _myrank >> step ^ 1;  // This toggles the lowest bit of the current step
      for (ISender *s = Senders->getFirst(); s != NULL; s = Senders->getNext()) {
	int rank = s->getRank();
	if (step==0) _allsenders[rank] = s;
	int testmask = rank >> step;
	if (testmask == mask) { // this tests if it is in the "other" half of the butterfly
	  _sendervectors[step].push_back(s);
	}
      }
      for (IReceiver *r = Receivers->getFirst(); r != NULL; r = Receivers->getNext()) {
	int rank = r->getRank(); 
	if (step==0) _allreceivers[rank] = r;
	int testmask = rank >> step;
	if (testmask == mask) {
	  ReceiverMap::iterator i=_receivermaps[step].find(rank);
	  assert(i==_receivermaps[step].end());
	  _receivermaps[step][rank] = r;	  
	}
      }
    }
  }

  /* * * Sending is easy - just run through pre-built list and send * * */

  void Send(int comstep) {
    SenderVector sv = _sendervectors[comstep];

#ifdef VERBOSE
    /* MPI_W begin: measure MPI send communiation + LENS marshalling */
    double now, then;
    now = MPI_Wtime();
#endif
    for (SenderVector::iterator s = sv.begin(); s != sv.end(); s++) {
      (*s)->send();
    }
#ifdef VERBOSE
    then = MPI_Wtime();
    /* MPI_W end: measure MPI send communiation + LENS marshalling */
    marshallPlusSendElapsed += (then-now);
#endif
  }

  /* * * Receiving is a little harder because may get data for next phase of butterfly * * */

  void Receive(int comstep) {
    // get list of receivers for this step
    ReceiverMap *rv = &_receivermaps[comstep]; 
    int sourcerank;                            

    // find how many are waiting for data (some may have already finished)
    int npending = 0;
    for (ReceiverMap::iterator r = rv->begin(); r != rv->end(); r++) {
      if (!(*r).second->done()) {
	npending++;
      }
    }

    // while some in the current list are still pending, receive and process
    while (npending) {
      int nrec;

      // receive from anyone - maps directly to MPIRecv.
      GlobalReceive(_recbuffer, BUFFERSIZE, &nrec, &sourcerank);

      // immediately dispatch using single list of all my receivers - not just the ones for this step
#ifdef VERBOSE
      /* MPI_W begin: measure LENS demarshalling */
      double now, then;
      now=MPI_Wtime();
#endif
      _allreceivers[sourcerank]->receive(_recbuffer, nrec);
#ifdef VERBOSE
      then=MPI_Wtime();
      demarshallElapsed += (then-now);
      /* MPI_W end: measure LENS demarshalling */
#endif
      // if the receiver is in the list for this step and it is done, one less to wait for
      if (_allreceivers[sourcerank]->done() && rv->find(sourcerank) != rv->end()) npending--;
    }
  }


  void GlobalReceive(char* _recbuffer, const int BUFFSIZE, int* nrec, int* sourcerank) {
#ifdef VERBOSE
    /* MPI_W begin: measure MPI receive communication */
    double now, then;
    now=MPI_Wtime();
#endif
    MPI_Status status;
    if (MPI_Recv(_recbuffer, BUFFSIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status) !=
	MPI_SUCCESS) {
      std::cerr << "MPI_Recv failed" << std::endl;
    }
    MPI_Get_count(&status, MPI_CHAR, nrec);
    *sourcerank = status.MPI_SOURCE;
#ifdef VERBOSE
    then=MPI_Wtime();
    receiveElapsed += (then-now);
    /* MPI_W end: measure MPI receive communication */
#endif
  };

  ~CommunicationEngine() {
    if (_sendervectors) delete [] _sendervectors;
    if (_receivermaps) delete [] _receivermaps;

    if (_wsndCounts) delete [] _wsndCounts;
    if (_wsndDispls) delete [] _wsndDispls;
    if (_wrcvCounts) delete [] _wrcvCounts;
    if (_wrcvDispls) delete [] _wrcvDispls;

    if (_vsbuff) delete [] _vsbuff;
    if (_vrbuff) delete [] _vrbuff;
    delete _vOutputStream;

    std::map<std::string, MPI_Datatype*>::iterator iter=_wsndTypes.begin(), end=_wsndTypes.end();
    MPI_Datatype* typePtr;
    for (; iter!=end; ++iter) {
      typePtr = iter->second;
      for (int i=0; i<_nprocs; ++i) {
	MPI_Type_free(&typePtr[i]);
      }
      delete [] (iter->second);
    }
    iter=_wrcvTypes.begin(), end=_wrcvTypes.end();
    for (; iter!=end; ++iter) {
      typePtr = iter->second;
      for (int i=0; i<_nprocs; ++i) {
	MPI_Type_free(&typePtr[i]);
      }
      delete [] (iter->second);
    }

    std::map<std::string, Args*>::iterator iter2=_vsndArgs.begin(), end2=_vsndArgs.end();
    for (; iter2!=end2; ++iter2)
      delete (iter2->second);
    iter2=_vrcvArgs.begin(), end2=_vrcvArgs.end();
    for (; iter2!=end2; ++iter2)
      delete (iter2->second);

    std::map<std::string, MPI_Comm>::iterator iter3=_phaseCommunicators.begin(), end3=_phaseCommunicators.end();
    for (; iter3!=end3; ++iter3) {
      MPI_Comm_free(&(iter3->second));
    }

#ifdef MEMCPY_MARSHALL
    std::map<std::string, MemPatternPointers>::iterator miter, 
      mend=_sndPatternMap.end();
    for (miter=_sndPatternMap.begin(); miter!=mend; ++miter) {
      delete [] miter->second._memPatterns;
    }
    mend=_rcvPatternMap.end();
    for (miter=_rcvPatternMap.begin(); miter!=mend; ++miter) {
      delete [] miter->second._memPatterns;
    }
#endif
  }

#ifdef VERBOSE
  static double demarshallElapsed;
  static double marshallPlusSendElapsed;
  static double receiveElapsed;
  static double collectivesElapsed;
  static std::map<std::string, double> collectivesElapsedMap;
#endif

 protected:
  int _nprocs;
  int _nsteps;
  int _myrank;
  char _recbuffer[BUFFERSIZE];
  ReceiverMap *_receivermaps;
  ReceiverMap _allreceivers;
  SenderVector *_sendervectors;
  SenderMap _allsenders;
  int P2P_TAG;

  int* _wsndCounts;
  int* _wsndDispls;
  int* _wrcvCounts;
  int* _wrcvDispls;
  std::map<std::string, MPI_Datatype*> _wsndTypes;
  std::map<std::string, MPI_Datatype*> _wrcvTypes;
  std::map<std::string, Args*> _vsndArgs;
  std::map<std::string, Args*> _vrcvArgs;
  std::map<std::string, MPI_Comm> _phaseCommunicators;
#ifdef MEMCPY_MARSHALL
  std::map<std::string, MemPatternPointers> _sndPatternMap;
  std::map<std::string, MemPatternPointers> _rcvPatternMap;
#endif
  OutputStream* _vOutputStream;
  char* _vsbuff;
  char* _vrbuff;

  Simulation *_sim;
};

#endif
#endif
