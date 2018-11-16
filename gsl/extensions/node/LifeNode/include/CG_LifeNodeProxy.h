// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef CG_LifeNodeProxy_H
#define CG_LifeNodeProxy_H

#include "Lens.h"
#ifdef HAVE_MPI
#include "CG_LifeNodeOutAttrPSet.h"
#include "CG_LifeNodeProxyDemarshaller.h"
#include "CG_LifeNodeSharedMembers.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "DemarshallerInstance.h"
#include "IntArrayDataItem.h"
#include "IntDataItem.h"
#include "NodeProxyBase.h"
#include "ShallowArray.h"
#include "String.h"
#include "ValueProducer.h"
#include <cassert>
#include <iostream>
#include <memory>
//#include "CG_LifeNodeCompCategory.h"

class CG_LifeNodeCompCategory;
class Constant;
class Edge;
class Node;
class Publisher;
class Variable;
//class CG_LifeNodeProxy;

class CG_LifeNodeProxy : public ValueProducer, public NodeProxyBase
{
   friend class CG_LifeNodePublisher;
   friend class CG_LifeNodeCompCategory;
   friend class LifeNodeCompCategory;
   private:

   class PhaseDemarshaller_FLUSH_LENS : public CG_LifeNodeProxyDemarshaller
   {
   public:
      PhaseDemarshaller_FLUSH_LENS()
      : CG_LifeNodeProxyDemarshaller(), publicValueDemarshaller()
      {
         _demarshallers.push_back(&publicValueDemarshaller);
      }
      PhaseDemarshaller_FLUSH_LENS(CG_LifeNodeProxy* proxy);
      /*
       //TUAN NOTE: we have to move to source file as we use CG_LifeNodeCompCategory's method
       //otherwise: incomplete type 'error' 
      PhaseDemarshaller_FLUSH_LENS(CG_LifeNodeProxy* proxy)
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->_demarshallerMap[demarshaller_index].um_publicValue[proxy->index]))
      : CG_LifeNodeProxyDemarshaller(proxy)
     // , 
     //    publicValueDemarshaller(&(
     //             (proxy->
     //              getCompCategory()->
     //              getDemarshaller(demarshaller_index)).um_publicValue[proxy->index]))
    #elif PROXY_ALLOCATION == OPTION_4
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->publicValue))
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->um_publicValue[proxy->index]))
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->proxy_um_publicValue[proxy->index]))
      : CG_LifeNodeProxyDemarshaller(proxy)
     // , publicValueDemarshaller(&(proxy->getCompCategory()->proxy_um_publicValue[proxy->index]))
      ////TODO fix here using above
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->index))
   #endif
#else
      : CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->publicValue))
#endif
      {
    #if PROXY_ALLOCATION == OPTION_3
         publicValueDemarshaller.setDestination(&(
                  (proxy->getCompCategory()->getDemarshaller(demarshaller_index)).um_publicValue[proxy->index]));
    #elif PROXY_ALLOCATION == OPTION_4
            publicValueDemarshaller.setDestination(&(proxy->getCompCategory()->proxy_um_publicValue[proxy->index]))
   #endif
         _demarshallers.push_back(&publicValueDemarshaller);
      }
      */
      void setDestination(CG_LifeNodeProxy *proxy);
      /*
      void setDestination(CG_LifeNodeProxy *proxy)
      {
         _proxy = proxy;
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
         //publicValueDemarshaller.setDestination(&(_proxy->_container->um_publicValue[_proxy->index]));
      ////TODO fix here using above
         //publicValueDemarshaller.setDestination(&(_proxy->_container->_demarshallerMap[demarshaller_index].um_publicValue[proxy->index]));
         publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->getDemarshallerMap()[demarshaller_index].um_publicValue[proxy->index]));
    #elif PROXY_ALLOCATION == OPTION_4
         //publicValueDemarshaller.setDestination(&(_proxy->_container->proxy_um_publicValue[proxy->index]));
         publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->proxy_um_publicValue[proxy->index]));
    #endif
#else
         publicValueDemarshaller.setDestination(&(_proxy->publicValue));
#endif
         reset();
      }
      */
      virtual ~PhaseDemarshaller_FLUSH_LENS()
      {
      }
   private:
      DemarshallerInstance< int > publicValueDemarshaller;
   };



   class PhaseDemarshaller_copy : public CG_LifeNodeProxyDemarshaller
   {
   public:
      PhaseDemarshaller_copy()
      : CG_LifeNodeProxyDemarshaller(), publicValueDemarshaller()
      {
         _demarshallers.push_back(&publicValueDemarshaller);
      }
      PhaseDemarshaller_copy(CG_LifeNodeProxy* proxy);
      /*
      PhaseDemarshaller_copy(CG_LifeNodeProxy* proxy)
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->_demarshallerMap[demarshaller_index].um_publicValue[proxy->index]))
      : CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&((proxy->getCompCategory()->getDemarshaller(demarshaller_index)).um_publicValue[proxy->index]))
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->publicValue))
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->um_publicValue[proxy->index]))
      //TODO fix here using above
    #elif PROXY_ALLOCATION == OPTION_4
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->proxy_um_publicValue[proxy->index]))
      : CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->getCompCategory()->proxy_um_publicValue[proxy->index]))
    #endif
#else
      : CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->publicValue))
#endif
      {
         _demarshallers.push_back(&publicValueDemarshaller);
      }
      */
      
      void setDestination(CG_LifeNodeProxy *proxy);
      /*
      void setDestination(CG_LifeNodeProxy *proxy)
      {
         _proxy = proxy;
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
         //publicValueDemarshaller.setDestination(&(_proxy->_container->um_publicValue[_proxy->index]));
      //TODO fix here using above
         //publicValueDemarshaller.setDestination(&(_proxy->_container->_demarshallerMap[demarshaller_index].um_publicValue[proxy->index]));
         publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->getDemarshallerMap()[demarshaller_index].um_publicValue[proxy->index]));
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->_demarshallerMap[demarshaller_index].um_publicValue[proxy->index]))
    #elif PROXY_ALLOCATION == OPTION_4
         //publicValueDemarshaller.setDestination(&(_proxy->_container->proxy_um_publicValue[proxy->index]));
         publicValueDemarshaller.setDestination(&(_proxy->getCompCategory()->proxy_um_publicValue[proxy->index]));
      //: CG_LifeNodeProxyDemarshaller(proxy), publicValueDemarshaller(&(proxy->_container->proxy_um_publicValue[proxy->index]))
    #endif
#else
         publicValueDemarshaller.setDestination(&(_proxy->publicValue));
#endif
         reset();
      }
      */
      virtual ~PhaseDemarshaller_copy()
      {
      }
   private:
      DemarshallerInstance< int > publicValueDemarshaller;
   };


   public:
      virtual int* CG_get_ValueProducer_value();
#ifdef HAVE_MPI
      static void CG_recv_copy_demarshaller(std::unique_ptr<CG_LifeNodeProxyDemarshaller> &ap);
#endif
#ifdef HAVE_MPI
      static void CG_recv_FLUSH_LENS_demarshaller(std::unique_ptr<CG_LifeNodeProxyDemarshaller> &ap);
#endif
      const CG_LifeNodeSharedMembers& getSharedMembers();
      virtual void addPostVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset);
      virtual void addPostEdge(Edge* CG_edge, ParameterSet* CG_pset);
      virtual void addPostNode(NodeDescriptor* CG_node, ParameterSet* CG_pset);
      CG_LifeNodeProxy();
      virtual ~CG_LifeNodeProxy();
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
      CG_LifeNodeCompCategory* getCompCategory() {return _container;};
      int getDemarshallerIndex() {return demarshaller_index;};
      int getDataIndex() {return index;}; //index local to the Demarshaller object
   #elif PROXY_ALLOCATION == OPTION_4
      CG_LifeNodeCompCategory* getCompCategory() {return _container;};
      int getDataIndex() {return index;}; //index global within the CG_xxxCompCategory 
   #endif
#endif
   protected:
#if defined(HAVE_GPU) 
    #if PROXY_ALLOCATION == OPTION_3
      int demarshaller_index;
      int index; //local to the given CCDermarshaller
      static CG_LifeNodeCompCategory* _container;

      void setCompCategory(int _index, CG_LifeNodeCompCategory* __container, int _demarshaller_index) {
         index = _index;
         _container = __container;
         demarshaller_index = _demarshaller_index; 
      }      //void setCCDemarshaller(int _index, CG_LifeNodeCompCategory* cg) { index = _index; _container=cg; }
   #elif PROXY_ALLOCATION == OPTION_4
      int index; //global index
      static CG_LifeNodeCompCategory* _container;
      void setCompCategory(int _index, CG_LifeNodeCompCategory* cg){
         index = _index;
         _container = cg;
      }
   #endif
#else
      int value;
      int publicValue;
      ShallowArray< int* > neighbors;
#endif
   private:
      CG_LifeNodeSharedMembers& getNonConstSharedMembers();
};

#endif
#endif
