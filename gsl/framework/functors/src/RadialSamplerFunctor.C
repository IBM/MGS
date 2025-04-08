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

#include "RadialSamplerFunctor.h"
#include "LensContext.h"
#include "ConnectionContext.h"
#include "DataItem.h"
#include "IntArrayDataItem.h"
#include "FloatArrayDataItem.h"
#include "NumericDataItem.h"
#include "FunctorType.h"
#include "GridLayerDescriptor.h"
#include "SurfaceOdometer.h"
#include "VolumeOdometer.h"
#include "Grid.h"
#include "Simulation.h"
#include "NodeDescriptor.h"
#include "NodeSet.h"
#include "NodeAccessor.h"
#include "InstanceFactoryQueriable.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include "rndm.h"

#include <sstream>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <utility>

//#define DEBUG_TIMER 

RadialSamplerFunctor::RadialSamplerFunctor()
: _responsibility(ConnectionContext::_BOTH), _refNode(0), _radius(0), 
  _borderTolerance(0), _direction(0), _currentNode(0), _nbrNodes(0)
{
  _square_radius = 0.0;
}

RadialSamplerFunctor::RadialSamplerFunctor(const RadialSamplerFunctor& rsf)
: _responsibility(rsf._responsibility), _refNode(rsf._refNode), _radius(rsf._radius), 
  _borderTolerance(rsf._borderTolerance), _direction(rsf._direction),
  _currentNode(rsf._currentNode), _nbrNodes(rsf._nbrNodes)
{
  _nodes=rsf._nodes;
  _refcoords=rsf._refcoords;
  _square_radius = _radius * _radius;
}

void RadialSamplerFunctor::duplicate(std::unique_ptr<Functor>&& fap) const
{
   fap.reset(new RadialSamplerFunctor(*this));
}


RadialSamplerFunctor::~RadialSamplerFunctor()
{
}


void RadialSamplerFunctor::doInitialize(LensContext *c, 
					const std::vector<DataItem*>& args)
{
  int nbrArgs=args.size();
  if (nbrArgs!=1 && nbrArgs !=2 && nbrArgs !=3) {
      std::ostringstream msg;
      msg << "RadialSampler: invalid arguments!" << std::endl
	  << "\texpected: RadialSampler(float radius) or" << std::endl
	  << "\texpected: RadialSampler(float radius, int borderTolerance)" << std::endl      
	  << "\texpected: RadialSampler(float radius, int borderTolerance, int direction)" 
	  << std::endl;
      throw SyntaxErrorException(msg.str());
  }
  NumericDataItem *radiusDI = dynamic_cast<NumericDataItem*>(args[0]);
  if (radiusDI==0) {
    std::ostringstream msg;
    msg << "RadialSampler: argument 1 is not a NumericDataItem" << std::endl
	<< "\texpected: RadialSampler(float radius)" << std::endl
	<< "\texpected: RadialSampler(float radius, int borderTolerance)" 
	<< std::endl;
    throw SyntaxErrorException(msg.str());
  }
  _radius=radiusDI->getFloat();
  _square_radius = _radius * _radius;
  if (nbrArgs==2) {
    NumericDataItem *borderToleranceDI = 
      dynamic_cast<NumericDataItem*>(args[1]);
    if (borderToleranceDI==0) {
      std::ostringstream msg;
      msg 
	<< "RadialSampler: argument 2 is not a NumericDataItem" 
	<< std::endl
	<< "\texpected: RadialSampler(float radius, int borderTolerance)."
	<< std::endl;
      throw SyntaxErrorException(msg.str());
    }
    _borderTolerance=unsigned(borderToleranceDI->getInt());
  }
  if (nbrArgs==3) {
    NumericDataItem *directionDI = 
      dynamic_cast<NumericDataItem*>(args[2]);
    if (directionDI==0) {
      std::ostringstream msg;
      msg 
        << "RadialSampler: argument 3 is not a NumericDataItem" 
        << std::endl
        << "\texpected: RadialSampler(float radius, int borderTolerance, int direction)."
        << std::endl;
      throw SyntaxErrorException(msg.str());
    }    
    _direction=unsigned(directionDI->getInt());
  }
}

void RadialSamplerFunctor::doExecute(LensContext *c, 
				     const std::vector<DataItem*>& args, 
				     std::unique_ptr<DataItem>& rvalue)
{
   ConnectionContext *cc = c->connectionContext;
#define REUSE_MEMORY
#ifdef REUSE_MEMORY
   ConnectionContext::Responsibility& resp = cc->current;
#else
   ConnectionContext::Responsibility resp = cc->current;
#endif
   bool refNodeDifferent = false;
   NodeSet* source=0;
   NodeDescriptor** slot=0;
#if defined(SUPPORT_MULTITHREAD_CONNECTION)
   std::vector<NodeDescriptor*>* slots;
#endif

   switch(resp) {
      case ConnectionContext::_SOURCE:
         //if (_speak) std::cout<<" each source node based on a complete sampling with a radius surrounding a ref node";
         source = cc->sourceSet;
#if defined(SUPPORT_MULTITHREAD_CONNECTION)
         cc->sourceNodes.resize(0);
         slots = &cc->sourceNodes;
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
         if (c->sim->isSimulatePass())
         {
           *slots = std::move(c->sim->ND_from_to[c->sim->_currentConnectNodeSet][cc->sourceRefNode].first);
           return;
         }
#endif
#else
         slot = &cc->sourceNode;
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
         if (c->sim->isSimulatePass())
         {
           auto& nodes = c->sim->ND_from_to[c->sim->_currentConnectNodeSet][cc->sourceRefNode];
           if (nodes.first.size() == nodes.second)
           {
             cc->done = true;
           }else
           {
             cc->done = false;
             *slot = nodes.first[nodes.second++];
           }
           return;
         }
#endif
#endif
         if(_refNode != cc->sourceRefNode) {
            _refNode = cc->sourceRefNode;
	    _refNode->getNodeCoords(_refcoords);
            refNodeDifferent = true;
         }
         break;
      case ConnectionContext::_DEST:
         //if (_speak) std::cout<<" each destination node based on a complete sampling within a radius surrounding a ref node";
         source = cc->destinationSet;
#if defined(SUPPORT_MULTITHREAD_CONNECTION)
         cc->destinationNodes.resize(0);
         slots = &cc->destinationNodes;
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
         if (c->sim->isSimulatePass())
         {
           *slots = std::move(c->sim->ND_from_to[c->sim->_currentConnectNodeSet][cc->destinationRefNode].first);
           return;
         }
#endif
#else
         slot = &cc->destinationNode;
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
         if (c->sim->isSimulatePass())
         {
           auto& nodes = c->sim->ND_from_to[c->sim->_currentConnectNodeSet][cc->destinationRefNode];
           if (nodes.first.size() == nodes.second)
           {
             cc->done = true;
           }else
           {
             cc->done = false;
             *slot = nodes.first[nodes.second++];
           }
           return;
         }
#endif
#endif
         if(_refNode != cc->destinationRefNode) {
            _refNode = cc->destinationRefNode;
	    _refNode->getNodeCoords(_refcoords);
            refNodeDifferent = true;
         }
         break;
      case ConnectionContext::_BOTH:
         throw SyntaxErrorException(
	    "RadialSamplerFunctor: invalid responsibility specification");
   }

    
#if defined(SUPPORT_MULTITHREAD_CONNECTION) && \
   SUPPORT_MULTITHREAD_CONNECTION == USE_ONLY_MAIN_THREAD
   if (cc->restart) {
     std::vector<int> coords, begincoords, endcoords, 
       mincoords, maxcoords, gridSize;
     mincoords = source->getBeginCoords();
     maxcoords = source->getEndCoords();
     gridSize =  source->getGrid()->getSize();
     int min, max, minTolerated, maxTolerated, absMax;
     for(unsigned i=0;i<_refcoords.size();++i) {
       min = _refcoords[i] - int(ceil(_radius));
       max = _refcoords[i] + int(ceil(_radius));
       minTolerated = mincoords[i]-_borderTolerance;
       maxTolerated = maxcoords[i]+_borderTolerance;
       absMax = gridSize[i]-1;

       min = (min<0)? 0:min;
       max = (max>absMax)? absMax:max;
       begincoords.push_back((min<minTolerated)? minTolerated:min);
       endcoords.push_back((max>maxTolerated)? maxTolerated:max);
     }

     NodeSet ns(*source);
     ns.setCoords(begincoords, endcoords);
     ns.getNodes(_nodes);
     _currentNode = 0;
     _nbrNodes = _nodes.size();
#ifdef DEBUG_TIMER 
     c->sim->benchmark_timelapsed_diff("... parallelConnection using sub-nodeset");
#endif
   }
   {
     // Pre loop
     const size_t nloop = _nbrNodes; 
     size_t offset_start = 0;
     size_t offset_end = nloop;

     // loop over all items
     for(auto i = offset_start;i<offset_end;i++)
     {
       auto _currentNode = i;

       float square_distance, dd;
       //bool outside=true;
       NodeDescriptor* n;
       std::vector<int> coords;
       n = _nodes[_currentNode];
       n->getNodeCoords(coords);     
       if (
           (_direction == 0) || // both direction
           ((_direction > 0) && // positive direction
            ((coords[0] >= _refcoords[0])
             && (coords[1] >= _refcoords[1])
             && (coords[2] >= _refcoords[2]))) ||
           ((_direction < 0) && // negative direction
            ((coords[0] <= _refcoords[0])
             && (coords[1] <= _refcoords[1])
             && (coords[2] <= _refcoords[2])))
          )
       {
         square_distance = 0;
         for(unsigned i=0;i<coords.size();++i) {
           dd = _refcoords[i] - coords[i];
           square_distance += dd*dd;
         }
         //distance=sqrt(distance);
       }
       else
         square_distance = _square_radius + 1.0;

       if (square_distance<=_square_radius) {
         slots->push_back(_nodes[_currentNode]);
       }
     }
     // Post loop
     //std::cout << ".. completed multi-thread with " << slots->size() << " node found\n"; 
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
     c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode].first = *slots;
#endif
     _currentNode = _nbrNodes;
   }

#elif defined(SUPPORT_MULTITHREAD_CONNECTION) && \
  (SUPPORT_MULTITHREAD_CONNECTION == USE_DYNAMIC_THREADPOOL || \
  SUPPORT_MULTITHREAD_CONNECTION == USE_STATIC_THREADPOOL) 

#if defined(USING_SUB_NODESET)
   if (cc->restart) {
     std::vector<int> coords, begincoords, endcoords, 
       mincoords, maxcoords, gridSize;
     mincoords = source->getBeginCoords();
     maxcoords = source->getEndCoords();
     gridSize =  source->getGrid()->getSize();
     int min, max, minTolerated, maxTolerated, absMax;
     for(unsigned i=0;i<_refcoords.size();++i) {
       min = _refcoords[i] - int(ceil(_radius));
       max = _refcoords[i] + int(ceil(_radius));
       minTolerated = mincoords[i]-_borderTolerance;
       maxTolerated = maxcoords[i]+_borderTolerance;
       absMax = gridSize[i]-1;

       min = (min<0)? 0:min;
       max = (max>absMax)? absMax:max;
       begincoords.push_back((min<minTolerated)? minTolerated:min);
       endcoords.push_back((max>maxTolerated)? maxTolerated:max);
     }

     NodeSet ns(*source);
     ns.setCoords(begincoords, endcoords);
     ns.getNodes(_nodes);
     _currentNode = 0;
     _nbrNodes = _nodes.size();
#ifdef DEBUG_TIMER 
     c->sim->benchmark_timelapsed_diff("... parallelConnection using sub-nodeset");
#endif
   }
#else
   if (cc->restart) {
     //get all  the nodes in nodeset
     source->getNodes(_nodes);
     _currentNode = 0;
     _nbrNodes = _nodes.size();
      std::cout << ".... load all the nodes RadialSampler"<<std::endl;
   }
#endif
    //const size_t nthreads = std::thread::hardware_concurrency();
    //TUAN NOTE: currently only maximum 10 threads

#if defined(USE_THREADPOOL_C11)
#ifdef DEBUG_TIMER 
     c->sim->benchmark_timelapsed_diff("... staticMTConnection");
#endif
  //std::unique_ptr<ThreadPoolC11> threadPoolC11;
    //const size_t nthreads = std::min(10, (int)std::thread::hardware_concurrency());
    size_t nthreads = c->sim->threadPoolC11->size();
    //c->sim->threadPoolC11->restart();
    {
      // Pre loop
      //std::vector<std::thread> threads(nthreads);
      std::mutex critical;
      const size_t nloop = _nbrNodes; 
      size_t offset_start = 0;
      size_t offset_end = 0;
      ///size_t portion = std::min((size_t)1, (size_t)(nloop/nthreads));
      //int min_portion_size = 100; //elements
      int min_portion_size = 20; //elements
      //size_t portion = std::max((size_t)min_portion_size, (size_t)(nloop/nthreads));
//#define USE_ALSO_MAINTHREAD
#if defined(USE_ALSO_MAINTHREAD)
      size_t portion = std::max((size_t)min_portion_size, (size_t)(nloop/(nthreads+1)));
#else
      size_t portion = std::max((size_t)min_portion_size, (size_t)(nloop/nthreads));
#endif
      int t;
      //std::cout<<"parallel ("<<nthreads<<" threads): " << nloop << " elements "<<std::endl;
      //std::cout << ".... start threadPools"<<std::endl;
#if defined(USE_ALSO_MAINTHREAD)
      size_t last_offset_for_pool = nloop - portion;
#else
      size_t last_offset_for_pool = nloop;
#endif
      for(t = 0; t<nthreads and offset_end <  last_offset_for_pool;t++)
      {
        /* each thread bind to a lambdas function
             the lambdas function accepts 3 arguments: 
               t= thread index
               bi = start index of data
               ei = end index of data
         */
        //c->sim->threadPoolC11->enqueue(
        offset_start = t*portion;
#if defined(USE_ALSO_MAINTHREAD)
        offset_end = std::min(last_offset_for_pool, (t+1)*portion);
#else
        offset_end = std::min(nloop, (t+1)*portion);
#endif
        c->sim->threadPoolC11->submit(
            [this, &slots, &critical](const int bi, const int ei, const int t)
            {
              // loop over all items
              for(int i = bi;i<ei;i++)
              {
                auto currentNode = i;

                float distance, dd;
                //bool outside=true;
                NodeDescriptor* n;
                std::vector<int> coords;
                n = _nodes[currentNode];
                n->getNodeCoords(coords);     
                // inner loop
                {
                 if (
                     (_direction == 0) || // both direction
                     ((_direction > 0) && // positive direction
                      ((coords[0] >= _refcoords[0])
                       && (coords[1] >= _refcoords[1])
                       && (coords[2] >= _refcoords[2]))) ||
                     ((_direction < 0) && // negative direction
                      ((coords[0] <= _refcoords[0])
                       && (coords[1] <= _refcoords[1])
                       && (coords[2] <= _refcoords[2])))
                     )
                   {
                     distance = 0;
                     for(unsigned i=0;i<coords.size();++i) {
                       dd = _refcoords[i] - coords[i];
                       distance += dd*dd;
                     }
                     distance=sqrt(distance);
                   }
                 else
                   distance = _radius + 1.0;
   
                 if (distance<=_radius) {
                   // make update critical
                   std::lock_guard<std::mutex> lock(critical);
                   slots->push_back(_nodes[currentNode]);
                 }
                }
              }
            }, offset_start, offset_end, t
        );
        //std::cout << ".. enqueued ("<< t+1 <<"-th task):"<<std::endl;
      }
#if defined(USE_ALSO_MAINTHREAD)
      t++;
      offset_start = t*portion;
      offset_end = std::min(last_offset_for_pool, (t+1)*portion);
      [this, &slots, &critical](const int bi, const int ei, const int t)
      {
        // loop over all items
        for(int i = bi;i<ei;i++)
        {
          auto currentNode = i;

          float distance, dd;
          //bool outside=true;
          NodeDescriptor* n;
          std::vector<int> coords;
          n = _nodes[currentNode];
          n->getNodeCoords(coords);     
          // inner loop
          {
           if (
               (_direction == 0) || // both direction
               ((_direction > 0) && // positive direction
                ((coords[0] >= _refcoords[0])
                 && (coords[1] >= _refcoords[1])
                 && (coords[2] >= _refcoords[2]))) ||
               ((_direction < 0) && // negative direction
                ((coords[0] <= _refcoords[0])
                 && (coords[1] <= _refcoords[1])
                 && (coords[2] <= _refcoords[2])))
               )
             {
               distance = 0;
               for(unsigned i=0;i<coords.size();++i) {
                 dd = _refcoords[i] - coords[i];
                 distance += dd*dd;
               }
               distance=sqrt(distance);
             }
           else
             distance = _radius + 1.0;

           if (distance<=_radius) {
             // make update critical
             std::lock_guard<std::mutex> lock(critical);
             slots->push_back(_nodes[currentNode]);
           }
          }
        }
      }( offset_start, offset_end, t);
#endif
      //c->sim->threadPoolC11->processQueue();
      //std::cout << ".. we only use ("<< t <<" threads):"<<std::endl;
      //c->sim->threadPoolC11->waitFinished(); //until queue is empty
      while ( ! c->sim->threadPoolC11->finishedJobs()) {}; //until jobs completed 
      // Post loop
      //std::cout << ".. RadialSampler completed multi-thread with " << slots->size() << " node found\n"; 
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
       c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode].first = *slots;
#endif
      _currentNode = _nbrNodes;
    }
#else
#ifdef DEBUG_TIMER 
     c->sim->benchmark_timelapsed_diff("... dynamicMTConnection");
#endif
    {
      // Pre loop
      //std::cout<<"parallel ("<<nthreads<<" threads):"<<std::endl;
      std::mutex critical;
      const size_t nloop = _nbrNodes; 
      size_t offset_start = 0;
      size_t offset_end = 0;
      ///size_t portion = std::min((size_t)1, (size_t)(nloop/nthreads));
      int min_portion_size = 100; //elements
      //int min_portion_size = 3; //elements
      //int min_portion_size = 2; //elements
      int max_nthreads_can_use = 10;
      const size_t nthreads = 
        std::min((size_t)std::min(max_nthreads_can_use, (int)std::thread::hardware_concurrency()),
            (size_t)std::ceil(((double)nloop)/min_portion_size));
      std::vector<std::thread> threads(nthreads);
      //size_t portion = std::max((size_t)min_portion_size, (size_t)(nloop/nthreads));
//#if defined(USE_ALSO_MAINTHREAD)
//      size_t portion = std::max((size_t)min_portion_size, (size_t)(nloop/(nthreads+1)));
//#else
      size_t portion = std::max((size_t)min_portion_size, (size_t)(nloop/nthreads));
//#endif
      int t;
      for(int t = 0;t<nthreads and offset_end < nloop;t++)
      {
        /* each thread bind to a lambdas function
             the lambdas function accepts 3 arguments: 
               t= thread index
               bi = start index of data
               ei = end index of data
         */
        offset_start = t*portion;
//#if defined(USE_ALSO_MAINTHREAD)
//        offset_end = std::min(last_offset_for_pool, (t+1)*portion);
//#else
        offset_end = std::min(nloop, (t+1)*portion);
//#endif
        threads[t] = std::thread(std::bind(
              [&](const int bi, const int ei, const int t)
              {
                // loop over all items
                for(int i = bi;i<ei;i++)
                {
                  auto _currentNode = i;

                  float distance, dd;
                  //bool outside=true;
                  NodeDescriptor* n;
                  std::vector<int> coords;
                  n = _nodes[_currentNode];
                  n->getNodeCoords(coords);     
                  // inner loop
                  {
                   if (
                       (_direction == 0) || // both direction
                       ((_direction > 0) && // positive direction
                        ((coords[0] >= _refcoords[0])
                         && (coords[1] >= _refcoords[1])
                         && (coords[2] >= _refcoords[2]))) ||
                       ((_direction < 0) && // negative direction
                        ((coords[0] <= _refcoords[0])
                         && (coords[1] <= _refcoords[1])
                         && (coords[2] <= _refcoords[2])))
                       )
                     {
                       distance = 0;
                       for(unsigned i=0;i<coords.size();++i) {
                         dd = _refcoords[i] - coords[i];
                         distance += dd*dd;
                       }
                       distance=sqrt(distance);
                     }
                   else
                     distance = _radius + 1.0;
     
                   if (distance<=_radius) {
                     // make update critical
                     std::lock_guard<std::mutex> lock(critical);
                     slots->push_back(_nodes[_currentNode]);
                   }
                  }
                }
              }, offset_start, offset_end, t));
      }
      std::for_each(threads.begin(),threads.end(),[](std::thread& x){x.join();});
      // Post loop
      //std::cout << ".. completed multi-thread with " << slots->size() << " node found\n"; 
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
       c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode].first = *slots;
#endif
      _currentNode = _nbrNodes;
    }
#endif

#else
   if (cc->restart) {
     /* get the subset of nodes (based on coord index), 
      *  ... before testing each of them using '_radius' later */
     std::vector<int> coords, begincoords, endcoords, 
       mincoords, maxcoords, gridSize;
     mincoords = source->getBeginCoords();
     maxcoords = source->getEndCoords();
     gridSize =  source->getGrid()->getSize();
     int min, max, minTolerated, maxTolerated, absMax;
     for(unsigned i=0;i<_refcoords.size();++i) {
       min = _refcoords[i] - int(ceil(_radius));
       max = _refcoords[i] + int(ceil(_radius));
       minTolerated = mincoords[i]-_borderTolerance;
       maxTolerated = maxcoords[i]+_borderTolerance;
       absMax = gridSize[i]-1;

       min = (min<0)? 0:min;
       max = (max>absMax)? absMax:max;
       begincoords.push_back((min<minTolerated)? minTolerated:min);
       endcoords.push_back((max>maxTolerated)? maxTolerated:max);
     }

     NodeSet ns(*source);
     ns.setCoords(begincoords, endcoords);
     /* this may take the longest time */
     ns.getNodes(_nodes);
     _currentNode = 0;
     _nbrNodes = _nodes.size();
#ifdef DEBUG_TIMER 
     c->sim->benchmark_timelapsed_diff("... linearConnection");
#endif
   }

   if (_currentNode==_nbrNodes) {
     *slot = 0;
     cc->done = true;
     return;
   }
   
   float distance, dd;
   bool outside=true;
   NodeDescriptor* n;
   std::vector<int> coords;
   while (outside) {
     n = _nodes[_currentNode];
     n->getNodeCoords(coords);     
     if (
         (_direction == 0) || // both direction
         ((_direction > 0) && // positive direction
          ((coords[0] >= _refcoords[0])
           && (coords[1] >= _refcoords[1])
           && (coords[2] >= _refcoords[2]))) ||
         ((_direction < 0) && // negative direction
          ((coords[0] <= _refcoords[0])
           && (coords[1] <= _refcoords[1])
           && (coords[2] <= _refcoords[2])))
         )
       {
         
         distance = 0;
         for(unsigned i=0;i<coords.size();++i) {
           dd = _refcoords[i] - coords[i];
           distance += dd*dd;
         }
         distance=sqrt(distance);

       }
     else
       distance = _radius + 1.0;
     
     if (distance<=_radius) {
       outside=false;
       *slot = _nodes[_currentNode];
#if defined(REUSE_NODEACCESSORS) and defined(REUSE_EXTRACTED_NODESET_FOR_CONNECTION)
       c->sim->ND_from_to[c->sim->_currentConnectNodeSet][_refNode].first.push_back(*slot);
#endif
      }
     else if (++_currentNode==_nbrNodes) {
       *slot = 0;
       cc->done = true;
       return;
     }
   }
   ++_currentNode;
   cc->done = false;
#endif
}
