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

#ifndef TEXTUSERINTERFACE_H
#define TEXTUSERINTERFACE_H
#ifndef DISABLE_PTHREADS
#include "Copyright.h"
#include <pthread.h>

#include "UserInterface.h"
#include "Simulation.h"

#include <string>
#include <list>

class TextUserInterface : public UserInterface
{
   public:
      TextUserInterface();
      void getUserInput(Simulation& sim);
      std::string getCommand();
      ~TextUserInterface();
   private:
      // mutex to protect changes to 'state' variable
      pthread_mutex_t _stateMutex;
      pthread_cond_t _inputGotten;
      std::string _user_input;
      std::list<std::string> _command;
};

inline TextUserInterface::TextUserInterface()
{
   pthread_mutex_init(&_stateMutex, NULL);
   pthread_cond_init(&_inputGotten, NULL);
   _user_input = "";
}

inline TextUserInterface::~TextUserInterface()
{
}

inline std::string TextUserInterface::getCommand()
{
   pthread_mutex_lock(&_stateMutex);
   while (_command.size() == 0)
      pthread_cond_wait(&_inputGotten, &_stateMutex);
   std::string tmp = _command.back();
   _command.pop_back();
   pthread_mutex_unlock(&_stateMutex);
   return tmp;
}

inline void TextUserInterface::getUserInput(Simulation& sim)
{
   int i = 1;
   while(i) {
      char c[256];
//      std::cin.getline(c,256);
      int s=scanf("%s", c);
      _user_input.assign(c);
      pthread_mutex_lock(&_stateMutex);
      _command.push_front(_user_input);
      pthread_mutex_unlock(&_stateMutex);
      pthread_cond_broadcast(&_inputGotten);
      if ((_user_input == "QUIT") || (_user_input == "quit")) {
         i = 0;
         sim.stop();
	 pthread_exit((void*) 0);
      }
      if ((_user_input == "PAUSE") || (_user_input == "pause")) {
         std::cout<<"Simulation paused at iteration number "<<sim.getIteration()<<"." << std::endl ;
         sim.pause();            //issue a pause
      }
      if ((_user_input == "RESUME") || (_user_input == "resume")) {
         sim.resume();
      }
   }                             //of while
}                                // of get_user_input

#endif
#endif
