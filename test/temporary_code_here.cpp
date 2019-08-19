
#ifdef THREAD_POOL

void ConnectNodeSetsFunctor::userExecuteThreadPool(LensContext* CG_c, NodeSet*& source, NodeSet*& destination, Functor*& sampling, Functor*& sourceOutAttr, Functor*& destinationInAttr) 
{
//#define DEBUG
//#ifdef DEBUG
//     if (CG_c->sim->getRank()==0)
//     {
//       CG_c->sim->benchmark_timelapsed(".. ConnectNodeSetsFunctor (userExecute() start)");
//     } 
//     CG_c->sim->resetCounter();
//#endif
//   CG_c->connectionContext->reset();
//   ConnectionContext* cc = CG_c->connectionContext;
//   
//   std::unique_ptr<ThreadPool> _threadsPool;
//   if (CG_c->sim->isGranuleMapperPass()) {
//      //may return 0 when not able to detect
//      unsigned concurentThreadsSupported = std::thread::hardware_concurrency();
//      if (concurentThreadsSupported != 1 and concurentThreadsSupported != 0) {
//         //Create a thread pool with the desired number of threads
//         auto _numCpus = sysconf(_SC_NPROCESSORS_ONLN);
//         auto _numThreads = concurentThreadsSupported;
//         bool bindThreadsToCpus = true;
//         _threadPool = new ThreadPool(_numThreads, _numCpus, bindThreadsToCpus);
//      }
//   }
//   // source  is partitioned into k partitions
//   // dest    is partitioned into k partitions
//   source.setNumPartitions(concurentThreadsSupported);
//   dest.setNumPartitions(concurentThreadsSupported);
//
//   std::deque<WorkUnit*> _workUnits;
//    for (x = 0; x < k; x++)
//    {
//       for (i = 0; i < k; i++)
//       {
//	  j = (i+x) mod k;
//	 put into the queue (userConnect [source_i, dest_j])
//       }
//       //process by ThreadPool
//       if (it->getWorkUnits().size() > 0) {
//       _threadPool->processQueue(it->getWorkUnits());
//       }
//    }
//
//   cc->sourceSet = source;
//   cc->destinationSet = destination;
//
//   std::vector<DataItem*> nullArgs;
//   std::unique_ptr<DataItem> outAttrRVal;
//   std::unique_ptr<DataItem> inAttrRVal;
//   std::unique_ptr<DataItem> rval;
//
//   // call sampfctr2, which will set source and destination nodes 
//   // (and maybe other stuff)
//   sampling->execute(CG_c, nullArgs, rval);
//
//   // loop until one of the nodes is null
//   //while(cc->destinationNode!=0 && cc->sourceNode!=0)
//
//   Connector* lc;
//
//   if (CG_c->sim->isGranuleMapperPass()) {
//     lc=_noConnector;
//   } else if (CG_c->sim->isCostAggregationPass()) {
//     lc=_granuleConnector;
//   } else if (CG_c->sim->isSimulatePass()) {
//     lc=_lensConnector;
//   } else {
//     std::cerr<<"Error, ConnectNodeSetsFunctor : no connection context set!"<<std::endl;
//     exit(0);
//   }
//
//#if defined(SUPPORT_MULTITHREAD_CONNECTION)
//   while(!cc->done) {
//      cc->restart = false;
//      for (int i = 0; i < cc->sourceNodes.size(); ++i)
//      {
//	 cc->currentSample++;
//	 cc->sourceNode = cc->sourceNodes[i];
//
//	 ParameterSetDataItem *psdi;
//	 sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
//	 psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
//	 if (psdi==0) {
//	    throw SyntaxErrorException(
//		  "ConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
//	 }
//	 cc->outAttrPSet = psdi->getParameterSet();
//
//	 destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
//	 psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
//	 if (psdi==0) {
//	    throw SyntaxErrorException(
//		  "ConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
//	 }
//	 cc->inAttrPSet = psdi->getParameterSet();
//
//#ifdef DEBUG
//	 CG_c->sim->increaseCounter();
//#endif
//	 lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
//	       cc->inAttrPSet, CG_c->sim);
//      }
//
//      // call sampfctr2, which will set source and destination nodes 
//      // (and maybe other stuff)
//      sampling->execute(CG_c, nullArgs, rval);
//   }
//#else
//   while(!cc->done) {
//      cc->restart = false;
//      cc->currentSample++;
//
//      ParameterSetDataItem *psdi;
//      sourceOutAttr->execute(CG_c, nullArgs, outAttrRVal);
//      psdi = dynamic_cast<ParameterSetDataItem*>(outAttrRVal.get());
//      if (psdi==0) {
//         throw SyntaxErrorException(
//	    "ConnectNodeSets: OutAttrPSet functor did not return a Parameter Set!");
//      }
//      cc->outAttrPSet = psdi->getParameterSet();
//
//      destinationInAttr->execute(CG_c, nullArgs, inAttrRVal);
//      psdi = dynamic_cast<ParameterSetDataItem*>(inAttrRVal.get());
//      if (psdi==0) {
//         throw SyntaxErrorException(
//	    "ConnectNodeSets: InAttrPSet functor did not return a Parameter Set!");
//      }
//      cc->inAttrPSet = psdi->getParameterSet();
//
//#ifdef DEBUG
//      CG_c->sim->increaseCounter();
//#endif
//      lc->nodeToNode(cc->sourceNode, cc->outAttrPSet, cc->destinationNode, 
//		     cc->inAttrPSet, CG_c->sim);
//
//      // call sampfctr2, which will set source and destination nodes 
//      // (and maybe other stuff)
//      sampling->execute(CG_c, nullArgs, rval);
//   }
//#endif
//#ifdef DEBUG
//     if (CG_c->sim->getRank()==0)
//     {
//	std::string msg;
//	if (CG_c->sim->isGranuleMapperPass()) {
//	   msg = "_noConnector";
//	} else if (CG_c->sim->isCostAggregationPass()) {
//	   msg = "_granuleConnector";
//	} else if (CG_c->sim->isSimulatePass()) {
//	   msg = "_lensConnector";
//	} 
//	std::cout << ".........." << msg << std::endl;
//       CG_c->sim->benchmark_timelapsed(".. ConnectNodeSetsFunctor (userExecute() end)");
//	std::cout << ".......... nodeToNode() has been called " << CG_c->sim->getCounter() << " times"<< std::endl;
//     } 
//#endif
}
#endif

