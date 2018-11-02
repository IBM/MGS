// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif 

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>

#include "SimInitializer.h"
#include "CommunicationEngine.h"
#include "CommandLine.h"
#ifdef VERBOSE
#include "MpiAsynchSender.h"
#endif

#ifndef DISABLE_PTHREADS
#include "GraphicalUserInterface.h"
#include "TextUserInterface.h"
#include "Browser.h"
#include "SBrowser.h"
#endif // DISABLE_PTHREADS

#include "Simulation.h"
#include "Repertoire.h"
#include "LensLexer.h"
#include "LensContext.h"
#include "PauseActionable.h"
#include "TriggeredPauseAction.h"
#include "SyntaxErrorException.h"
#include "OneToOnePartitioner.h"
//#define CART_COORDS
#ifdef CART_COORDS
#include "BGCartesianPartitioner.h"
#endif
#include "ReadGraphPartitioner.h"
//#include "ConnectionIncrement.h"
#ifdef USING_BLUEGENE
#include "BG_AvailableMemory.h"
#endif

extern int lensparse(void*);
extern int yydebug;

#ifndef DISABLE_PTHREADS

extern "C"
{
   void* UI_thread_function(void *arg) {
      Simulation *sim_ptr = (Simulation *) arg;
      UserInterface* this_ui = sim_ptr->getUI();

      this_ui->getUserInput(*sim_ptr);
      void* p=0;
      return (p);
   }
}

#endif // DISABLE_PTHREADS

#ifdef VERBOSE
double MpiAsynchSender::sendElapsed=0;
double CommunicationEngine::demarshallElapsed=0;
double CommunicationEngine::marshallPlusSendElapsed=0;
double CommunicationEngine::receiveElapsed=0;
double CommunicationEngine::collectivesElapsed=0;
std::map<std::string, double> CommunicationEngine::collectivesElapsedMap;
#endif

RunTimeTopology SimInitializer::_topology;

SimInitializer::SimInitializer()
  : _rank(0), _size(1)
{
}

bool SimInitializer::execute(int* argc, char*** argv)
{
   bool retVal = true;

#ifdef HAVE_MPI
   // !!! Important don't comment this out!
   MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &_size);

#endif 
   retVal = internalExecute(*argc, *argv);
   return retVal;
}

bool SimInitializer::internalExecute(int argc, char** argv)
{
   bool verbose=false;
   if (_rank==0) verbose=true;
   CommandLine commandLine(verbose);
   if (!(commandLine.parse(argc, argv))) {
      return false;
   }
   std::unique_ptr<Simulation> sim;
   std::unique_ptr<LensLexer> scanner;
   std::unique_ptr<LensContext> context;
   
   char const* infilename = commandLine.getGslFile().c_str();
   std::istream *infile;
   std::ostream *outfile = &std::cout;

   // Debugging flag for bison
   yydebug = 0;

#ifndef PROFILING
   bool processed=false;
#ifdef USING_BLUEGENE
   char temporaryName[256];
   std::ostringstream os;
   os<<infilename<<"ps";
   strcpy(temporaryName,os.str().c_str());
   std::string line;
   std::ifstream topf("Topology.h");
   std::string top[3] = {"-1", "-1", "-1"};
   if (topf.is_open()) {
     while ( topf.good() ) {
       getline(topf, line);
       std::stringstream ss(line);
       std::string tok;
       ss>>tok;
       if (tok=="#define") {
	 ss>>tok;
	 if (tok=="_X_") ss>>top[0];
	 else if (tok=="_Y_") ss>>top[1];
	 else if (tok=="_Z_") ss>>top[2];
	 else {
	   std::cerr<<"Error reading Topology.h!"<<std::endl;
	   exit(-1);
	 }
       }
     }
     topf.close();
     if (_rank==0) {
       std::ifstream inf(infilename);
       if (inf.is_open()) {
	 std::ofstream outf(temporaryName);
	 if (outf.is_open()) {
	   while ( inf.good() ) {
	     getline(inf, line);
	     std::size_t pos=0;
	     pos=line.find("_X_",0);
	     while(pos!=std::string::npos) {
	       line.replace(pos, 3, top[0]);
	       pos=line.find("_X_",pos+1);
	     }
	     pos=line.find("_Y_",0);
	     while(pos!=std::string::npos) {
	       line.replace(pos, 3, top[1]);
	       pos=line.find("_Y_",pos+1);
	     }
	     pos=line.find("_Z_",0);
	     while(pos!=std::string::npos) {
	       line.replace(pos, 3, top[2]);
	       pos=line.find("_Z_",pos+1);
	     }
	     outf<<line<<std::endl;
	   }
	   outf.close();
	   inf.close();
	   processed=true;
	 }
       }
     }
     else processed=true;
   }
   MPI_Barrier(MPI_COMM_WORLD);
#else
   char temporaryName[256] = "/tmp/bc_mpp.XXXXXX";               // modified by Jizhu Lu on 01/10/2006
   char command[256];
   if (mkstemp(temporaryName)) {                                 // added by Jizhu Lu on 01/10/2006
#ifdef LINUX
     sprintf(command,"cpp %s %s", infilename, temporaryName);
#endif
#ifdef AIX
     sprintf(command,"/usr/gnu/bin/gcpp %s %s", infilename, temporaryName);
#endif
     int s=system(command);
     processed=true;
   }
#endif /* USING_BLUEGENE */
   if (processed) 
     infile = new std::ifstream(temporaryName);
   else {
     std::cerr << "Unable to preprocess gsl file, aborting..." << std::endl << std::endl;
     return false;
   }
#else
   infile = new std::ifstream(infilename);
#endif /* PROFILING */
   
#ifndef DISABLE_PTHREADS
   //sim.reset(new Simulation(commandLine.getThreads(), commandLine.getBindCpus(), commandLine.getWorkUnits(), commandLine.getSeed()));
   sim.reset(new Simulation(commandLine.getThreads(), commandLine.getBindCpus(), commandLine.getWorkUnits(), commandLine.getSeed(), commandLine.getGpuID()));
#else // DISABLE_PTHREADS
   //sim.reset(new Simulation(commandLine.getWorkUnits(), commandLine.getSeed());
   sim.reset(new Simulation(commandLine.getWorkUnits(), commandLine.getSeed(), commandLine.getGpuID());
#endif // DISABLE_PTHREADS

   if (!commandLine.getEnableErd()) sim->disableEdgeRelationalData();

   Partitioner* partitioner=0;
   std::ostringstream fname;
   fname<<"LENS.gph";
   bool outputGraph=false;
   if (sim->getRank()==0 && commandLine.getOutputGraph()) outputGraph=true;

   if (commandLine.getReadGraph()) partitioner = new ReadGraphPartitioner(fname.str().c_str(), outputGraph);
   else {
#ifdef HAVE_MPI
#ifdef CART_COORDS
     partitioner = new BGCartesianPartitioner(fname.str().c_str(), outputGraph, sim.get(), 0, 0, 0);
#else
     partitioner = new OneToOnePartitioner(fname.str().c_str(), outputGraph);
#endif
#else   
     partitioner = new OneToOnePartitioner(fname.str().c_str(), outputGraph);
#endif
   }
   sim->setPartitioner(partitioner);
   if (partitioner->requiresCostAggregation()) sim->setCostAggregationPass();

   scanner.reset(new LensLexer(infile,outfile));
   context.reset(new LensContext(sim.get()));
   context->lexer = scanner.get();

   lensparse(context.get());

   
   // If there was an error return w/o starting...Trivial error
   if (context->isError()) {
      if (sim->getRank()==0) std::cerr << "Quitting due to errors..." << std::endl << std::endl;
      return false;
   }
   /*
   try {
   */
#if 0
     // read Resource Requirements from file "Resource.txt"
     std::ifstream resourceFile("Resource.txt");

     std::string buffer;
     std::ostringstream ostr;
     char ch;
     while (resourceFile.get(ch))ostr << ch;
     buffer = ostr.str();
     std::istringstream istr(buffer);

     ConnectionIncrement* computeCost;

     istr >> computeCost->_computationTime;
     istr >> computeCost->_memoryBytes;
     istr >> computeCost->_communicationBytes;

     sim->setComputeCost(computeCost);

     std::cerr << "_connectionIncrement->_computationTime=" << _connectionIncrement->_computationTime << std::endl;
     std::cerr << "_connectionIncrement->_memoryBytes=" << _connectionIncrement->_memoryBytes << std::endl;
     std::cerr << "_connectionIncrement->_communicationBytes=" << _connectionIncrement->_communicationBytes << std::endl;
#endif

     std::unique_ptr<LensContext> firstPassContext;
     context->duplicate(firstPassContext);
     firstPassContext->addCurrentRepertoire(sim->getRootRepertoire());
     if (sim->getRank()==0) printf("\nThe first execution of the parse tree begins.\n\n");
     if (sim->getRank()==0 && sim->isCostAggregationPass()) std::cout << "Aggregating costs in granule graph." << std::endl << std::endl;

     firstPassContext->execute();
     if (firstPassContext->isError()) {
       if (sim->getRank()==0) printf("Quitting due to errors...\n\n");
       return false;
     }
     if (sim->getRank()==0) printf("Partitioning simulation.\n\n");
     sim->setSeparationGranules();
     sim->setGraph();
#ifdef USING_BLUEGENE
     if (sim->getRank()==0) printf("Available memory after simulation initialization first pass: %lf MB.\n\n",AvailableMemory());
#endif 
     
     if (commandLine.getSimulate()) {
       if (sim->getRank()==0) printf("Resetting simulation.\n\n");
       sim->resetInternals();
       sim->setSimulatePass();
       context->addCurrentRepertoire(sim->getRootRepertoire());
       if (sim->getRank()==0) printf("The second execution of the parse tree begins.\n\n");
       context->execute();
     }
   /*
   }
   catch (SyntaxErrorException& e) {
     e.printError();
     return false;
   }
   */
   // If there was an error return /wo starting...High level error
   if (context->isError()) {
     if (sim->getRank()==0) std::cerr << "Quitting due to errors..." << std::endl << std::endl;
     return false;
   }
   
   bool retVal = true;
   
   if (commandLine.getSimulate()) {
#ifdef USING_BLUEGENE
     if (sim->getRank()==0) printf("\nAvailable memory after simulation initialization second pass: %lf MB.\n\n", AvailableMemory());
#endif 
     if (sim->getRank()==0) printf("Initializing simulation.\n\n");
     retVal = runSimulationAndUI(commandLine, sim);
     if (sim->getRank()==0) printf("Simulation complete.\n\n");
   }
   else if (sim->getRank()==0) printf("Simulation suppressed.\n\n");

   delete infile;
   delete partitioner;

   unlink(temporaryName);

#ifdef VERBOSE
   if (sim->P2P()) {
     std::cerr<<MpiAsynchSender::sendElapsed;
#ifndef USING_BLUEGENE
     std::cerr<<std::endl;
#endif
     std::cerr<<CommunicationEngine::marshallPlusSendElapsed-MpiAsynchSender::sendElapsed;
#ifndef USING_BLUEGENE
     std::cerr<<std::endl;
#endif
     std::cerr<<CommunicationEngine::receiveElapsed;
#ifndef USING_BLUEGENE
     std::cerr<<std::endl;
#endif
     std::cerr<<CommunicationEngine::demarshallElapsed;
#ifndef USING_BLUEGENE
     std::cerr<<std::endl;
#endif
   }
   if (sim->AllToAllW() || sim->AllToAllV()) {
     std::map<std::string, double>::iterator iter, end=CommunicationEngine::collectivesElapsedMap.end();
     for (iter=CommunicationEngine::collectivesElapsedMap.begin(); iter!=end; ++iter) {
     	std::ostringstream os;
	os<<iter->first<<" "<<iter->second;
#ifndef USING_BLUEGENE
	os<<std::endl;
#endif
        std::cerr<<os.str();
     }
     std::ostringstream os;
     os<<"Total "<<CommunicationEngine::collectivesElapsed;
#ifndef USING_BLUEGENE
     os<<std::endl;
#endif
     std::cerr<<os.str();
   }
#endif
   return retVal;
}

bool SimInitializer::runSimulationAndUI(
   CommandLine& commandLine, std::unique_ptr<Simulation>& sim)
{
#ifndef DISABLE_PTHREADS
   bool noUI = false;
   std::unique_ptr<GraphicalUserInterface> guiInterface;
   std::unique_ptr<TextUserInterface> textInterface;

   if (commandLine.getUserInterface() == "gui") {
      guiInterface.reset(
	 new GraphicalUserInterface(commandLine.getGuiPort(), *sim));
      assert(guiInterface.get());

      std::unique_ptr<PauseActionable> pa(new SBrowser(*sim, 
						     guiInterface.get()));
      sim->getTriggeredPauseAction()->insert(pa);
      sim->setUI(guiInterface.get());
      if (sim->getRank()==0) std::cout << "Graphical user interface initialized." << std::endl << std::endl;
   }
   else if (commandLine.getUserInterface() == "text") {
      textInterface.reset(new TextUserInterface);
      assert(textInterface.get());

      std::unique_ptr<PauseActionable> pa(new Browser(*sim, 
						    textInterface.get()));
      sim->getTriggeredPauseAction()->insert(pa);
      sim->setUI(textInterface.get());
      if (sim->getRank()==0) std::cout << "Text user interface initialized." << std::endl << std::endl;
   } else  {
      // Internal error check
      assert(commandLine.getUserInterface() == "none");
      noUI = true;
      if (sim->getRank()==0) printf("No user interface present.\n\n");
   }
   if (noUI) {
      sim->start();
   } else {
      int error;
      pthread_t uiThread;   
      
      if((error = pthread_create(
	     &uiThread, NULL, UI_thread_function, (void*)sim.get())) != 0) {
	 switch(error) {
         case EAGAIN: 
	    std::cerr << "Thread creation error, EAGAIN" << std::endl << std::endl;
	    break;
         case EINVAL: 
	    std::cerr << "Thread creation error, EINVAL" << std::endl << std::endl;
	    break;
         case ENOMEM: 
	    std::cerr << "Thread creation error, ENOMEM" << std::endl << std::endl;
	    break;
	 }
	 return false;
      }

      bool needToDetachUI;
      needToDetachUI = sim->start();

      if (needToDetachUI) {
	 pthread_cancel(uiThread);
      } else {
	 pthread_join(uiThread, NULL);      
      }
   }

#else 
   sim->start();

#endif // DISABLE_PTHREADS
   return true;
}
