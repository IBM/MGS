// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "MgsServer.h"
#include "Simulation.h"
#include "NumericDataItem.h"
#include "UnsignedIntDataItem.h"
#include "UnsignedTrigger.h"
#include "ServiceDataItem.h"
#include "CustomStringDataItem.h"
#include "IntDataItem.h"
#include "PhaseDataItem.h"
#include "RuntimePhase.h"
#include "Publisher.h"
#include "Service.h"
#include "Repertoire.h"
#include "TriggerType.h"
#include "TypeRegistry.h"
#include "Grid.h"
#include "GridLayerDescriptor.h"
#include "ConnectionSet.h"
#include "InstanceFactoryRegistry.h"
#include "InstanceFactory.h"
#include "DataItem.h"
#include "GslContext.h"
#include "GenericService.h"
#include "SyntaxErrorException.h"
#include "Trigger.h"
#include "NDPairList.h"
#include <typeinfo>
#include <vector>
#include <list>
#include <utility>
#include <memory>
#include <string>
#include <sstream>
#include <cassert>

#ifdef AIX
MgsServer::MgsServer(int p, Simulation& s)
   : _connected(0), _reconnect(0), _s(s), _serveBuff(0), _controlBuff(0)
{
   Publisher* pubptr = s.getPublisher();
   Service* sptr = pubptr->getService("Iteration");

   GenericService<unsigned>* iterService;
   iterService = dynamic_cast<GenericService<unsigned>* >(sptr);
   assert(iterService != 0);
   // modification required to run in distributed computing environment -- Jizhu Lu on 03/29/2006
   assert(iterService != NULL);
   _simIteration = iterService->getData();

   StringDataItem desc = StringDataItem();
   desc.setString("Reports Every 50 Iterations to Server");

   IntDataItem delay = IntDataItem();
   delay.setInt(0);

   StringDataItem op = StringDataItem();
   op.setString("!%");

   UnsignedIntDataItem crit = UnsignedIntDataItem();
   crit.setUnsignedInt(50);

   ServiceDataItem svrc = ServiceDataItem();
   svrc.setService(sptr);

   std::unique_ptr<Phase> phaseAp(
      new RuntimePhase(_s.getFinalRuntimePhaseName()));
   PhaseDataItem phase(phaseAp);

   std::vector<DataItem*> vecky;
   vecky.push_back(&desc);
   vecky.push_back(&svrc);
   vecky.push_back(&op);
   vecky.push_back(&crit);
   vecky.push_back(&delay);
   vecky.push_back(&phase);

   std::unique_ptr<NDPairList> ndp(0);
   _itrTrigger = _s.getTriggerType("UnsignedTrigger")->getTrigger(vecky);
   addTrigger(_itrTrigger, "event", ndp);
//   _itrTrigger->addTriggerable(this);
   _portNum = p;
   _serveBuff = new char[MAXBUFFSIZE];
   _controlBuff = new char[MAXBUFFSIZE];
}

MgsServer::~MgsServer()
{
   delete _serveBuff;
   delete _controlBuff;
   std::vector<DataItem*>::iterator iter = _dataItems.begin();
   std::vector<DataItem*>::iterator end = _dataItems.end();
   for (;iter!=end;++iter) delete (*iter);
}

int MgsServer::isConnected()
{
   return(_connected);
}

void MgsServer :: initSocket()
{
   int sockfd;
   struct sockaddr_in my_addr;
   struct sockaddr_in their_addr;
   int sin_size;
   int yes = 1;

   std::cout<<"MgsServer: Setting up Socket..."<< std::endl;

   if((sockfd = socket(AF_INET,SOCK_STREAM, 0))==-1) {
      perror("socket");
      exit(-1);
   }
   if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &yes,
		  sizeof(int)) == -1) {
      perror("SOL_SOCKET");
      exit(-1);
   }
   my_addr.sin_family = AF_INET;
   my_addr.sin_port = htons(_portNum);
   my_addr.sin_addr.s_addr=INADDR_ANY;
   bzero(&(my_addr.sin_zero), 8);

   std::cout <<"MgsServer: Binding to socket.."<< std::endl;

   if(bind (sockfd, (struct sockaddr*) &my_addr, sizeof(struct sockaddr))==1) {
      perror("bind");
      std::cerr<<"MgsServer: can't bind"<< std::endl;
      exit(-1);
   }

   std::cout <<"MgsServer: Listening to port (BACKLOG = " << BACKLOG << ")..." << std::endl;

   int listenValue = listen(sockfd, BACKLOG);

   std::cout << "MgsServer: Listen returned: " << listenValue << std::endl;

   if(listenValue == -1) {
      std::cerr<<"MgsServer: Error listening!"<< std::endl;
      exit(-1);
   }

   std::cout << "MgsServer: Listening for client connection" << std::endl;

   sin_size = sizeof(struct sockaddr_in);

   if( (_controlSock = accept(sockfd, (struct sockaddr*) &their_addr,(socklen_t*) &sin_size) )==-1) {
      std::cerr<<"MgsServer: Error accepting!" << std::endl;
      exit(-1);
   }

   std::cout<< "MgsServer: sending connect..."<< std::endl;
   sendMsg("control connect");

   ///////////////////////////////////////////////////////////
   // now display socket

   std::cout << "MgsServer: Accepting connection on sock2" << std::endl;
   if( (_displaySock = accept(sockfd, (struct sockaddr*) &their_addr,(socklen_t*) &sin_size) )==-1) {

      std::cerr<<"error accepting on display socket!" << std::endl;
      exit(-1);
   }

   std::cout<< "MgsServer: Sending display connect"<< std::endl;

   sendMsg("display connect");
   char buff2[MAXBUFFSIZE];
   std::cout <<"MgsServer: Waiting to hear back from client..." << std::endl;
   recvMsg(buff2,15, _displaySock);

   if( strcmp(buff2, "display connect") ==0 ) {
      std::cout << "MgsServer: Client successfully echoed connect..." << std::endl;
   }
   else {
      std::cout<< "MgsServer: Incorrect client echo : " << buff2 << std::endl;
   }
   _connected=1;

   if(_reconnect) {
      _s.pause();
      _dataItems.clear();
   }

   std::cout <<"MgsServer: Closing original socket file descriptors.." << std::endl;
   close(sockfd);
}

void MgsServer::initDisplay()
{
   sendToClient("display repertoires begin");
   char simbuff[MAXBUFFSIZE];
   strcpy(simbuff, "display repertoire<>simulation<>");
   strcat(simbuff, _s.getName().c_str());
   strcat(simbuff, "<>1<>0<>0<>");
   sendToClient(simbuff);
   repExpand(_s.getRootRepertoire());
   sendToClient("display repertoires end");

//    std::list<std::string> names = _s.getServiceTriggerNames();

//    std::list<std::string> :: iterator nitr = names.begin();

//    while(nitr!=names.end()) {
//       char buff[MAXBUFFSIZE];
//       std::string trigname = (*nitr);
//       strcpy(buff, "display triglist entry <>");
//       strcat(buff, trigname.c_str());
//       strcat(buff,"<>");
//       sendToClient(buff);
//       nitr++;
//    }

   std::ostringstream msgstream;
   msgstream << "display numcpu " << _s.getNumCPUs() << std::ends;
   char msgBuff[MAXBUFFSIZE];
   strcpy(msgBuff,msgstream.str().c_str());

   sendToClient(msgBuff);

   std::ostringstream threadstream;
   threadstream << "display numthreads "<< _s.getNumThreads() << std::ends;
   char msgBuff2[MAXBUFFSIZE];
   strcpy(msgBuff2,threadstream.str().c_str());

   sendToClient(msgBuff2);

   initComposer();

   if(_reconnect) {

      if(_s.getIteration()!=0) {
         sendToClient("display button_behavior reconnect");
      }
      else {
         sendToClient("display button_behavior newsim");
      }

      std::ostringstream msgstream;
      msgstream << "display itr " << _s.getIteration() << std::ends;
      char msgBuff[MAXBUFFSIZE];
      strcpy(msgBuff,msgstream.str().c_str());
      sendToClient(msgBuff);
      sendToClient("display browser wakeup");
   }
   else {
      sendToClient("display button_behavior newsim");
   }
}

void MgsServer:: initComposer()
{
   sendToClient("display composer clear registries");
   const std::vector<InstanceFactoryRegistry*>& ifrs= _s.getInstanceFactoryRegistries();
   int nbrFactories = ifrs.size();

   for(int i=0; i < nbrFactories; ++i) {
      std::string typeName = ifrs[i]->getTypeName();

      char composeBuff[MAXBUFFSIZE];
      strcpy(composeBuff, "display composer registry ");
      strcat(composeBuff, typeName.c_str());

      sendToClient(composeBuff);

      const std::list<InstanceFactory*> & ifs = ifrs[i]->getInstanceFactoryList();
      std::list<InstanceFactory* > :: const_iterator facItr= ifs.begin();
      std::list<InstanceFactory* > :: const_iterator facEnd= ifs.end();

      while(facItr != facEnd) {
         sendPDescription(typeName.c_str(), (*facItr)->getName().c_str(), (*facItr)->getParameterDescription());
         facItr++;
      }
   }
}

char* MgsServer::serve()
{
   recvMsg(_serveBuff, 4,_controlSock);
   int msgSize = atoi(_serveBuff);
   if(MAXBUFFSIZE < msgSize) {
      std::cerr << "MgsServer: buffer too small!" << std::endl;
      sendMsg("control msg too big");
      strcpy(_serveBuff, "buffer overflow");
   }
   else {
      sendMsg("control got size");
      recvMsg(_serveBuff, msgSize, _controlSock);
      const char* newMsg = msgHandler(_serveBuff);
      if ((strcmp(newMsg, "control disconnect")!=0) && isConnected()) {
         sendMsg(newMsg);
      }
   }
   std::cout <<"MgsServer:  Received: " << _serveBuff << std::endl;
   return _serveBuff;
}

void MgsServer:: sendComposerError(const char* facName, const char* expectedType,const char* param2,int i, int errType)
{
   std::cerr<<"MgsServer: Dynamic cast of expected user entered parameter to StringDataItem and NumericDataItem failed!"<<std::endl;
   std::ostringstream errstream;
   errstream <<"display extended_error <>The server has returned the following:<>";
   errstream << "Parameter #" << i+1 << " for the " << facName << " you're constructing is not matched ";
   errstream << "correctly. Please check the arguments and try again.\n" << std::endl;

   char temp[MAXBUFFSIZE];
   strcpy(temp, "The expected type was: ");
   strcat(temp, expectedType);

   if(errType ==1) {
      strcat(temp, ", which is not compatible with the user entered value: \" ");
      strcat(temp, param2);
      strcat(temp, "\".");
   }
   else {
      strcat(temp, ", which is not compatible with an instance of type: ");
      strcat(temp, param2);
      strcat(temp, ".");
   }
   muter(temp);

   char msgbuff[MAXBUFFSIZE];
   strcpy(msgbuff, errstream.str().c_str());
   strcat(msgbuff, temp);
   strcat(msgbuff, "<>");

   sendToClient(msgbuff);
}

const char* MgsServer:: composerMsgHandler(char* msg)
{
   char composeBuff[MAXBUFFSIZE];
   composeBuff[0]='\0';

   if(strncmp(msg, "compose loadtype",16)==0) {
      strtok(msg, " ");
      strtok(NULL, " ");
      char* bigType = strtok(NULL, " ");
      char* specType = strtok(NULL, " ");
      const char* result = loadType(bigType, specType);
      initComposer();
      return result;
   }

   if(strncmp(msg, "compose instance", 16)==0) {
      std::cout << "MgsServer: composerMsgHandler:  attempting to compose instance" << std::endl;
      strtok(msg, " ");
      strtok(NULL, " ");
      char* regIndex = strtok(NULL, " ");
      char* facName = strtok(NULL, " ");
      char* paramDescIndex = strtok(NULL, " ");
      char* nameEntered = strtok(NULL, "<>");
      int regIdx = atoi(regIndex);
      int paramDescIdx = atoi(paramDescIndex);

      const std::vector<InstanceFactoryRegistry*>& ifrs = _s.getInstanceFactoryRegistries();
      InstanceFactoryRegistry* ifr = ifrs[regIdx];
      InstanceFactory* ifc = ifr->getInstanceFactory(_s, *_s.getDependencyParser(), std::string(facName));
      std::vector<std::vector<std::pair<std::string, DataItem*> > > const & paramDescList = ifc->getParameterDescription();
      std::vector<std::pair<std::string, DataItem*> > const & paramDesc = paramDescList[paramDescIdx];
      std::vector<DataItem*> args;
      int nbrArgs = paramDesc.size();

      for (int i=0; i<nbrArgs; ++i) {
         char* paramType = strtok(NULL, "<>");
         char* value = strtok(NULL, "<>");
         std::unique_ptr<DataItem> apdi;
         if (strcmp(paramType, "string")==0) {
            paramDesc[i].second->duplicate(apdi);
            StringDataItem* sdi = dynamic_cast<StringDataItem*>(apdi.get());
            NumericDataItem* ndi = dynamic_cast<NumericDataItem*>(apdi.get());
            if (sdi == 0 && ndi == 0) {
               const char* expected_type = apdi->getType();
               sendComposerError(facName, expected_type,value,i,1);
               return("control composer bad parameter");
            }

            //std::cout <<"[###MgsServer###] I guess the dynamic cast went smoothly" << std::endl;
            unMuter(value);
            if (sdi) sdi->setString(std::string(value));
            if (ndi) ndi->setString(std::string(value));
            args.push_back(apdi.release());
         }
         else if (strcmp(paramType, "dataitem")==0) {
            int idx = atoi(value);
            const char* inputType = _dataItems[idx]->getType();
            const char* expectedType = paramDesc[i].second->getType();
            if(strcmp(inputType, expectedType)==0) {
               _dataItems[idx]->duplicate(apdi);
               args.push_back(apdi.release());
            }
            else {
               sendComposerError(facName, expectedType, inputType,i, 2);
               return("control composer bad parameter");
            }
         }
         else {
            std::cerr<<"MgsServer: composerMsgHandler: bad input,  paramType = "<<paramType<<std::endl;
            std::ostringstream errstream;
            errstream <<"display extended_error <>The server has returned the following:<>";
            errstream << "Parameter #" << i+1 << " for this " << facName <<" object is not matched ";
            errstream << "correctly. Please check the arguments again.<>" << std::endl;
            char msgbuff[MAXBUFFSIZE];
            strcpy(msgbuff, errstream.str().c_str());
            sendToClient(msgbuff);
            return("control composer bad parameter");
         }
      }
      GslContext c(&_s);
      std::unique_ptr<DataItem> apdi;
      ifc->getInstance(apdi, &args, &c);
      // sendToClient
      char sendbuff[MAXBUFFSIZE];
      strcpy(sendbuff, "display dataitem ");
      strcat(sendbuff, apdi->getType());
      strcat(sendbuff, " ");
      strcat(sendbuff, nameEntered);
      sendToClient(sendbuff);

      keepDataItem(apdi);
      std::vector<DataItem*>::iterator iter = args.begin();
      std::vector<DataItem*>::iterator end = args.end();
      for (;iter!=end;++iter) delete (*iter);
      return("control display instance made");
   }
   return("control display badmsg");
}

const char* MgsServer::loadType(char* bigType, char* specType)
{
   std::cout <<"MgsServer: loadType: getting instance factories" << std::endl;
   std::vector<InstanceFactoryRegistry*> const & v = _s.getInstanceFactoryRegistries();
   std::string regname = bigType;
   std::string tname = specType;
   InstanceFactory* f=0;

   std::cout << "MgsServer: loadType: loading a " << specType << " from the " << bigType << " factory!";
   for(unsigned i=0; i < v.size(); i++) {
      if ( v[i]->getTypeName() == regname) {
         f =v[i]->getInstanceFactory(_s, *_s.getDependencyParser(), tname);
      }
   }

   if(f==0) {
      std::cout <<"f=0" << std::endl;
      std::ostringstream o;
      o <<"display extended_error <>The Server returned the following:<>";
      o <<"The type: " << specType << " could not be loaded. It was not found in the ";
      o << bigType << " registry. Please check the name and try again. <>"<< std::endl;
      char errbuff[MAXBUFFSIZE];
      strcpy(errbuff, o.str().c_str());
      sendToClient(errbuff);
      return("control display not loaded");
   }
   else {
      return("control display loaded");
   }
}

char* MgsServer:: controlMsgHandler(char* msg)
{
   _controlBuff[0]='\0';
   std::ostringstream msgstream;

   // std::cout << "control msg: " << msg << std::endl;

   if (strcmp(msg, "start")==0) {
      _s.run();
      strcpy(_controlBuff, "control display Starting simulation");
   }
   else if(strcmp(msg, "pause")==0) {
      _s.pause();
      msgstream<<"control display Simulation paused at iteration number "<<_s.getIteration()<<"."<<std::ends;
      strcpy(_controlBuff, msgstream.str().c_str());
      std::ostringstream msgstream2;
      msgstream2 << "display itr " << _s.getIteration() << std::ends;
      char msgBuff[MAXBUFFSIZE];
      strcpy(msgBuff,msgstream2.str().c_str());
      sendToClient(msgBuff);
   }
   else if( (strcmp(msg, "resume")==0) || (strcmp(msg,"start")==0)) {
      _s.resume();
      strcpy(_controlBuff,"control display Resuming simulation");
   }
   else if(strcmp(msg, "stop")==0) {
      msgstream<<"control display Simulation stopped at iteration number "<<_s.getIteration()<<"."<<std::ends;
      strcpy(_controlBuff, msgstream.str().c_str());
      _s.stop();
      shutdown();
   }
   else if(strcmp(msg, "disconnect")==0) {
      std::cout << "MgsServer: Shutting down..." << std::endl;
      sendMsg("control disconnect");
      std::cout <<"MgsServer Sent disconnect reply to control client" <<std::endl;
      sendToClient("display disconnect");
      strcpy(_controlBuff,"control disconnect");

      disconnect();
   }
   else if(strncmp(msg, "checkpoint",10)==0) {

      std::cerr << "Checkpointing disabled." << std::endl;
      assert(false);

      // @TODO: Remove Checkpoint from the GUI
//       strtok(msg, "<>");
//       char* chkname = strtok(NULL, "<>");
//       char* filename = strtok(NULL, "<>");
//       Checkpointable* chk = _s.getCheckpointable(chkname);
//       if(chk!= NULL) {
//          chk->checkpoint(filename);
//          strcpy(_controlBuff, "control checkpointed");
//          strcat(_controlBuff, chkname);
//       }
//       else {
//          strcpy(_controlBuff,"control checkpoint error: bad chkname");
//          strcat(_controlBuff, chkname);
//       }
   }
   else if(strncmp(msg, "restore",7)==0) {

      std::cerr << "Restoring disabled." << std::endl;
      assert(false);

      // @TODO: Remove Checkpoint from the GUI
//       strtok(msg, "<>");
//       char* chkname = strtok(NULL, "<>");
//       char* filename = strtok(NULL, "<>");
//       Checkpointable* chk = _s.getCheckpointable(chkname);
//       if(chk!= NULL) {
//          chk->restore(filename);
//          strcpy(_controlBuff, "control restored");
//          strcat(_controlBuff, chkname);
//       }
//       else {
//          strcpy(_controlBuff, "control restore error: bad chkname");
//          strcat(_controlBuff, chkname);
//       }
   }
   else if(strncmp(msg,"usertrigger", 11)==0) {
      std::cerr << "User triggers are removed" 
		<< ", this is an error condition, quitting." << std::endl;
      exit(-1);
   }
   else strcpy(_controlBuff,"control display command not understood");
   return _controlBuff;
}

void MgsServer::disconnect()
{
   _reconnect=1;

   shutdown();
   initSocket();
   initDisplay();
}

const char* MgsServer:: msgHandler(char* msg)
{
   if(strncmp(msg, "disconnect", 10)==0)
      return("disconnect");
   if(strncmp(msg, "compose", 7)==0) {
      return(composerMsgHandler(msg));
   }
   if(strncmp(msg, "control", 7)==0 ) {
      if(strlen(msg)> 8)
         return( controlMsgHandler(msg + 8));
   }
   if(strncmp(msg, "browse", 6)==0)
      return("browse command received");
   return("command not understood");
}

void MgsServer:: recvMsg(char* msg, int size, int sockfd)
{
   int numread=0;
   do {
      numread+=recv(sockfd, msg+numread, size-numread, 0);
   } while(numread < size);
   msg[size] ='\0';
}

const char* MgsServer:: sendToClient(const char* msg)
{
   sendMsg(msg);
   char buff[MAXBUFFSIZE];
   recvMsg(buff, 8, _displaySock);
   if(strcmp(buff,"received")==0)
      return("received");
   else
      return("sending error");
}

void MgsServer::sendMsg(const char* msg)
{
   // std::cout << "MgsServer::sendMsg: " << msg << std::endl;
   //note, the java readline method is going to expect a newline (or the client will hang)

   if( (strncmp(msg,"control", 7)==0) || (strncmp(msg,"browse", 6)==0) ) {
      //     std::cout << "MgsServer Control Socket Sending: " << msg << std::endl;
      silencer(msg);
      // std::cout << "silenced: " << msg << std::endl;
      if( (strlen(msg) >=15 ) && (strncmp(msg,"control display",15) ==0 )) {
	    //chopping off "control"
	 std::string mesStr(msg+8);
	 mesStr += "\n";
	 socketSend(_controlSock, mesStr);
      } else {
	 std::string mesStr(msg);
	 mesStr += "\n";
	 socketSend(_controlSock, mesStr);
      }
   }
   else if(strncmp(msg, "display0",8)==0) {
      silencer(msg);
      std::string mesStr(msg);
      mesStr += "\n";
      socketSend(_displaySock, mesStr);
   }
   else if( (strncmp(msg,"display", 7)==0) ) {
      // std::cout << "MgsServer Display Socket Sending: " << msg << std::endl;
      silencer(msg);
      std::string mesStr(msg);
      mesStr += "\n";
      socketSend(_displaySock, mesStr);
   }
   else {
      std::cerr<<"MgsServer: msgHandler: don't know how to handle msg: " << msg << std::endl;
   }
   return;
}

void MgsServer::socketSend(int fd, std::string& msg)
{
   int size = msg.size();
   int sentSize;
   int totalSent = 0;
   const char * curPlace = msg.c_str();	 
   while (size != 0) {
      sentSize = send(fd, curPlace + totalSent, size, 0);
      if (sentSize == -1) {
	 std::cerr << "Error while MgsServer::socketSend, msg: " << msg << std::endl;
	 break;
      }
      size -= sentSize;
      totalSent += sentSize;
   }	 
}

void MgsServer::shutdown()
{
   _connected = 0;
   close(_controlSock);
   close(_displaySock);
   std::cout << "MgsServer: shutdown: sockets closed" << std::endl;
}

void MgsServer::event(Trigger* trigger, NDPairList* ndPairList)
{
   if(!isConnected()) {
      return;
   }
   std::ostringstream msgstream;

   msgstream << "display itr " << *_simIteration <<std::ends;
   char msgBuff[MAXBUFFSIZE];
   strcpy(msgBuff,msgstream.str().c_str());
   sendToClient(msgBuff);
}

void MgsServer::repExpand(Repertoire* r)
{
   std::string name = r->getName();

   std::ostringstream msgstream;

   std::list<Repertoire*> reps = r->getSubRepertoires();
   std::map<GridLayerDescriptor*, std::list <ConnectionSet*> > csMap = 
      r->getConnectionSetMap();
   std::map<GridLayerDescriptor*, std::list<ConnectionSet*> >::iterator 
      mapitr = csMap.begin();

   std::vector<std::string> csetnames;

   for(mapitr = csMap.begin(); mapitr != csMap.end(); mapitr++) {
      std::list<ConnectionSet*>& clist = (*mapitr).second;
      std::list<ConnectionSet*>:: iterator citr;

      for(citr = clist.begin(); citr != clist.end(); citr++) {
         std::string csetname = (*citr)->getName();
         char csetnamebuff[MAXBUFFSIZE];
         strcpy(csetnamebuff, csetname.c_str());
         csetnames.push_back(csetnamebuff);
      }
   }

   int csets = csetnames.size();
   char repType[MAXBUFFSIZE];

   int layers;

   if(r->isGridRepertoire()) {
      strcpy(repType, "gridrep");
      layers = r->getGrid()->getLayers().size();
   }
   else {
      strcpy(repType, "composite");
      layers = 0;
   }

   msgstream<<"display repertoire<>"<<repType<<"<>"<<name.c_str()<<"<>"<<reps.size()<<"<>"<<csets<<"<>"<<layers<<"<>"<<std::ends;
   char msgBuff[MAXBUFFSIZE];
   strcpy(msgBuff,msgstream.str().c_str());

   sendToClient(msgBuff);

   if(csets > 0) {

      for(unsigned cidx=0; cidx < csetnames.size();cidx++) {
         char csetbuff[MAXBUFFSIZE];
         strcpy(csetbuff,"display repertoire cset ");
         strcat(csetbuff,csetnames[cidx].c_str());
         sendToClient(csetbuff);
      }
   }

   if(r->isGridRepertoire()) {
      Grid* g= r->getGrid();
      std::vector<GridLayerDescriptor*> const & gld = g->getLayers();
      std::string gname =  g->getName();
      char gridbuff [MAXBUFFSIZE];
      strcpy(gridbuff, "display repertoire<>grid<>");
      strcat(gridbuff,gname.c_str());
      strcat(gridbuff, "<>");
      sendToClient(gridbuff);
      for(unsigned l=0; l < gld.size(); l++) {
         char layerbuff[MAXBUFFSIZE];
         strcpy(layerbuff, "display repertoire<>layer<>");
         strcat(layerbuff, gld[l]->getName().c_str());
         strcat(layerbuff, "<>");
         sendToClient(layerbuff);
      }

   }

   if(reps.empty())
      return;

   std::list<Repertoire*>::const_iterator itr = reps.begin();

   while(itr != reps.end()) {
      repExpand(*itr);
      itr++;
   }
}

void MgsServer::sendPDescription(const char* type1,const char* type2, std::vector< std::vector< std::pair< std::string, DataItem*> > > const & pd)
{
   for(unsigned i=0; i < pd.size(); i++) {
      std::vector< std::pair < std::string, DataItem*> > const& v= pd[i];

      char sendbuff[MAXBUFFSIZE];
      strcpy(sendbuff, "display composer pdesc <>");
      strcat(sendbuff,type1 );
      strcat(sendbuff, "<>");
      strcat(sendbuff, type2);
      strcat(sendbuff, "<>");

      for(unsigned j=0; j < v.size(); j++) {
         char firstBuff[MAXBUFFSIZE];
         char secondBuff[MAXBUFFSIZE];

         strcpy(firstBuff, v[j].first.c_str());
         muter(firstBuff);
         strcat(sendbuff, firstBuff);
         strcat(sendbuff, "<>");
         strcpy(secondBuff, v[j].second->getType());
         muter(secondBuff);
         strcat(sendbuff, secondBuff);
         strcat(sendbuff, "<>");
      }

      sendToClient(sendbuff);
   }
}

void MgsServer::silencer(const char* s)
{
  char* ss=const_cast<char*>(s);
  for(unsigned i=0; i < strlen(ss); i ++) {
    if( ss[i] == '\n') {
      ss[i]= '\b';
      //std::cout << "silencing someone..." << std::endl;
    }
  }
}

void MgsServer::muter(char* s)
{
   for(unsigned i=0; i < strlen(s); i ++) {
      if( s[i] == '<') {
         s[i]= '\f';
         //std::cout << "muting someone..." << std::endl;
      }

      if( s[i] == '>') {
         s[i]= '\a';
         //std::cout << "muting someone..." << std::endl;
      }

   }
}

void MgsServer::unMuter(char* s)
{
   for(unsigned i=0; i < strlen(s); i ++) {
      if( s[i] == '\f') {
         s[i]= '<';
         //std::cout << "unmuting someone..." << std::endl;
      }

      if( s[i] == '\a') {
         s[i]= '>';
         //std::cout << "unmuting someone..." << std::endl;
      }

   }
}

void MgsServer::keepDataItem(std::unique_ptr<DataItem> & diap)
{
   _dataItems.push_back(diap.release());
}

inline TriggerableBase::EventType MgsServer::createTriggerableCaller(
   const std::string& functionName, NDPairList* ndpList, 
   std::unique_ptr<TriggerableCaller>&& triggerableCaller)
{
   if (functionName != "event") {
      throw SyntaxErrorException(
	 functionName + 
	 " is not defined in MgsServer as a Triggerable function.");
   } 
   triggerableCaller.reset(
      new MgsServerEvent(ndpList, &MgsServer::event, this));
   return TriggerableBase::_SERIAL;
}
#endif
