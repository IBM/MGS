/* Class::printCopyright(os)*/
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
/* end Class::printCopyright(os)*/

/* Class::printBeginning(os)*/
#ifndef CG_LifeNode_H
#define CG_LifeNode_H
/* end Class::printBeginning(os)*/

/* ... part of Class::generateHeader( "CG_LifeNode" )*/
#include "Lens.h"

/* Class::printHeaders(_headers, os)  - 
 *  using _headers (std::set   of { _name = "CG_header_name", _macroCondition = {_name="" | "MPI"} })
 *      which were generated 
 *              via InterfaceImplementorBase::generateInstanceBase()
instance->addHeader(...)
 *  add all headers */
#include "CG_LifeNodeInAttrPSet.h"
#include "CG_LifeNodeOutAttrPSet.h"
#include "CG_LifeNodePSet.h"
#ifdef HAVE_MPI
#include "CG_LifeNodeProxyDemarshaller.h"
#endif
#include "CG_LifeNodePublisher.h"
#include "CG_LifeNodeSharedMembers.h"
#include "DataItem.h"
#include "DataItemArrayDataItem.h"
#include "IntArrayDataItem.h"
#include "IntDataItem.h"
#ifdef HAVE_MPI
#include "Marshall.h"
#endif
#include "NodeBase.h"
#ifdef HAVE_MPI
#include "OutputStream.h"
#endif
#include "Service.h"
#include "ShallowArray.h"
#include "String.h"
#include "SyntaxErrorException.h"
#include "ValueProducer.h"
#include "VariableDescriptor.h"
#include <iostream>
#include <memory>
/* end Class::printHeaders(_headers, os)  - add all headers */


/* Class::printClasses(os)  - add all class declaration
 *  using _classes  (std::set   of { _name = "CG_header_name", _macroCondition = {_name="" | "MPI"} })
 *      which were generated 
 *              via InterfaceImplementorBase::generateInstanceBase()
instance->addClass(...)
 *
 * */
class CG_LifeNodeCompCategory;
class ConnectionIncrement;
class Constant;
class Edge;
class Node;
class Variable;
class VariableDescriptor;

/* Class::generateClassDefinition(os) 
 which traverse 
     <IncludeClass>::iter -> getClassCode()
 * */
class CG_LifeNode 
/*   <BaseClass*>: iter   -> getConditional(); //e.g. #ifdef MPI
 *
 *
 *   NOTE: A Class can be a member of another class so 
 *     if (_memberClass)
 *     {... }
 */
: public ValueProducer, public NodeBase
{
   /*
    * _friendDeclarations.size()
    * <FriendDeclaration>::iter  -> getCodeString();
    */
   friend class CG_LifeNodePublisher;
#ifdef HAVE_MPI
   friend class CG_LifeNodeCompCategory;
#endif
   friend class LifeNodeCompCategory;

   /*
    * Class::printAccessMemberClass(AccessType::PUBLIC, "public", os)
    */
   public:
      virtual int* CG_get_ValueProducer_value();
#if defined(HAVE_GPU)
      void setCompCategory(int _index, CG_LifeNodeCompCategory* cg) { index = _index; _container=cg; }
#endif
#if defined(HAVE_GPU)
      CG_LifeNodeGPUCompCategory* getContainer()
      {
      return _container;
      }
#endif
      virtual const char* getServiceName(void* data) const;
      virtual const char* getServiceDescription(void* data) const;
      virtual Publisher* getPublisher();
      virtual void initialize(ParameterSet* CG_initPSet);
      ///TUAN NOTE: we cann't make them '__global__' if we use static data member '_container'
      //  This is ok, as we don't put data member inside class
      //virtual CUDA_CALLABLE void initialize(RNG& rng)=0;
      //virtual CUDA_CALLABLE void update(RNG& rng)=0;
      //virtual CUDA_CALLABLE void copy(RNG& rng)=0;
      virtual void initialize(RNG& rng)=0;
      virtual void update(RNG& rng)=0;
      virtual void copy(RNG& rng)=0;
      virtual void getInitializationParameterSet(std::unique_ptr<ParameterSet>& initPSet) const;
      virtual void getInAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) const;
      virtual void getOutAttrParameterSet(std::unique_ptr<ParameterSet>& CG_castedPSet) const;
      virtual void acceptService(Service* service, const std::string& name);
      const CG_LifeNodeSharedMembers& getSharedMembers() const;
      virtual void addPostVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset);
      virtual void addPostEdge(Edge* CG_edge, ParameterSet* CG_pset);
      virtual void addPostNode(NodeDescriptor* CG_node, ParameterSet* CG_pset);
      virtual void addPreConstant(Constant* CG_constant, ParameterSet* CG_pset);
      virtual void addPreVariable(VariableDescriptor* CG_variable, ParameterSet* CG_pset);
      virtual void addPreEdge(Edge* CG_edge, ParameterSet* CG_pset);
      virtual void addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset);
      //TUAN TODO : make all connection function return 'bool'
      //so that we can detect if a connection is successfull for memory allocation
      //virtual bool addPreNode(NodeDescriptor* CG_node, ParameterSet* CG_pset);
      virtual ConnectionIncrement* getComputeCost() const;
      CG_LifeNode();
      virtual ~CG_LifeNode();

   /*
    * Class::printAccessMemberClass(AccessType::PROTECTED , "public", os)
    */
   protected:
#ifdef HAVE_MPI
      void CG_send_copy(OutputStream* stream) const
      {
         MarshallerInstance<int > mi0;
#if defined(HAVE_GPU) 
         //TUAN TODO revise here
         //mi0.marshall(stream, _container->um_publicValue[index]);
#else
         mi0.marshall(stream, publicValue);
#endif
      }
#endif
#ifdef HAVE_MPI
      void CG_getSendType_copy(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const;
#endif
#ifdef HAVE_MPI
      void CG_send_FLUSH_LENS(OutputStream* stream) const;
#endif
#ifdef HAVE_MPI
      void CG_getSendType_FLUSH_LENS(std::vector<int>& blengths, std::vector<MPI_Aint>& blocs) const;
#endif
      virtual TriggerableBase::EventType createTriggerableCaller(const std::string& CG_triggerableFunctionName, NDPairList* CG_triggerableNdpList, std::unique_ptr<TriggerableCaller>& CG_triggerableCaller);


      /*
       * The data member is generated, using info added via
       *    Class::addAttributes(const MemberContainer<DataType>& members, ....)
       *       in that 'members' hold vector of 
       *              key=name of data member ('value', 'publicValue', 'neighbors'), 
       *              value=object (IntType, ...)
       */
#if defined(HAVE_GPU) 
   protected:
      int index; //the index in the array
      //TUAN TODO: write a method function to provide the access to '_container' 
      //    so that we don't make '_container' public
      // and also to avoid the error 'access incomplete type' if we make '_container' as private
      //
      //CG_LifeNodeCompCategory* _container;
      static CG_LifeNodeCompCategory* _container; //just declaration, need to be instantiated in the .cpp
      //LifeNodeCompCategory* _container;
      //int& value(){ return _container->um_value[index];}
      //inline int& publicValue(){ return _container->um_publicValue[index];}
      //inline ShallowArray_Flat<int*> & neighbors(){ return _container->um_neighbors[index];}
#else
   protected:
      int value;
      int publicValue;
      ShallowArray< int* > neighbors;
#endif

   /*
    * Class::printAccessMemberClass(AccessType::PRIVATE, "public", os)
    */
   private:
      CG_LifeNodeSharedMembers& getNonConstSharedMembers();
};

/* Class::printPartnerClasses(os) */

/* Class::printExternCDefinition(os) */

/* Class::printExternCPPDefinition(os) */

/* os << "#endif" */
#endif
