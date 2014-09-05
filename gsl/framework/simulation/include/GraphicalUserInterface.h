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

#ifndef GRAPHICALUSERINTERFACE_H
#define GRAPHICALUSERINTERFACE_H
#ifndef DISABLE_PTHREADS
#include "Copyright.h"

#include <pthread.h>
#include <sstream>
#include <string>
#include <list>
#include "UserInterface.h"
#include "LENSServer.h"
#include "Simulation.h"

class GraphicalUserInterface : public UserInterface
{
   public:
      GraphicalUserInterface(int guiPort, Simulation& sim);
      LENSServer* getServer();
      void getUserInput(Simulation& sim);
      std::string getCommand();
      ~GraphicalUserInterface();
   private:
      // mutex to protect changes to 'state' variable
      pthread_cond_t _inputGotten;
      pthread_mutex_t _userInputMutex;
      pthread_mutex_t _commandMutex;

      LENSServer* _server;
      std::string _user_input;
      std::list<std::string> _command;
      int _guiPort;
};

inline GraphicalUserInterface::GraphicalUserInterface(int guiPort, Simulation& sim)
: _server(0)
{
   pthread_mutex_init(&_commandMutex, NULL);
   pthread_mutex_init(&_userInputMutex, NULL);
   pthread_cond_init(&_inputGotten, NULL);
   _user_input = "";
   _guiPort = guiPort;
   std::cout<<"GUI waiting for port setting from Browser..."<<std::endl;
   _server = new LENSServer(_guiPort,sim);
   _server->initSocket();
}

inline LENSServer* GraphicalUserInterface::getServer()
{
   return (_server);
}

inline std::string GraphicalUserInterface::getCommand()
{
   pthread_mutex_lock(&_commandMutex);
   while (_command.size() == 0)
      pthread_cond_wait(&_inputGotten, &_commandMutex);
   std::string tmp = _command.back();
   _command.pop_back();
   pthread_mutex_unlock(&_commandMutex);
   return tmp;
}

inline void GraphicalUserInterface::getUserInput(Simulation& sim)
{
   int i = 1;
   _server->initDisplay();
   while(i) {
      _user_input.assign(_server->serve());
      pthread_mutex_lock(&_commandMutex);
      _command.push_front(_user_input);
      pthread_mutex_unlock(&_commandMutex);
      pthread_cond_broadcast(&_inputGotten);

      if (_user_input == "control disconnect") {
         std::cout <<"I see the disconnect command" << std::endl;
         //i = 0;
      }

      if(_user_input == "control stop") {
         std::cout << "seeing stop command! " << std::endl;
         i=0;
	 // Done, terminate
         sim.stop();
	 pthread_exit((void*) 0);

      }
   }
   //  _server->shutdown();
   //of while
}                                // of get_user_input

inline GraphicalUserInterface::~GraphicalUserInterface()
{
   delete _server;
}
#endif
#endif
