// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "Edge.h"
#include "SharedCCBase.h"
#include "Generatable.h"
#include "EdgeConnection.h"
#include "AccessType.h"
#include "BaseClass.h"
#include "Class.h"
#include "ConstructorMethod.h"
#include "Method.h"
#include "CustomAttribute.h"
#include "Attribute.h"
#include "Constants.h"
#include <memory>
#include <cassert>
#include <set>
#include <string>
#include <sstream>
#include <stdio.h>
#include <string.h>

Edge::Edge(const std::string& fileName) 
   : SharedCCBase(fileName), _preNode(0), _postNode(0) 
{
}

Edge::Edge(const Edge& rv)
   : SharedCCBase(rv), _preNode(0), _postNode(0) 
{
   copyOwnedHeap(rv);
}

Edge Edge::operator=(const Edge& rv)
{
   if (this != &rv) {
      SharedCCBase::operator=(rv);
      destructOwnedHeap();
      copyOwnedHeap(rv);
   }
   return *this;
}


void Edge::duplicate(std::unique_ptr<Generatable>&& rv) const
{
   rv.reset(new Edge(*this));
}

std::string Edge::getType() const
{
   return "Edge";
}

std::string Edge::generateExtra() const
{
   std::ostringstream os;
   os << SharedCCBase::generateExtra();

   if (_preNode) {
      os << "\n" << _preNode->getString();
      if (_postNode) {
	 os << "\n";
      }
   }
   if (_postNode) {
      if (_preNode == 0) {
	 os << "\n";
      }
      os << _postNode->getString();
   }

   return os.str();
}

EdgeConnection* Edge::getPreNode() 
{
   return _preNode;
}

void Edge::setPreNode(std::unique_ptr<EdgeConnection>&& con) 
{
   delete _preNode;
   _preNode = con.release();
}

EdgeConnection* Edge::getPostNode() 
{
   return _postNode;
}

void Edge::setPostNode(std::unique_ptr<EdgeConnection>&& con) 
{
   delete _postNode;
   _postNode = con.release();
}

Edge::~Edge() 
{
   destructOwnedHeap();
}

void Edge::copyOwnedHeap(const Edge& rv)
{
   if(rv._preNode) {
      std::unique_ptr<EdgeConnection> dup;
      rv._preNode->duplicate(std::move(dup));
      _preNode = dup.release();
   } else {
      _preNode = 0;
   }
   if(rv._postNode) {
      std::unique_ptr<EdgeConnection> dup;
      rv._postNode->duplicate(std::move(dup));
      _postNode = dup.release();
   } else {
      _postNode = 0;
   }
}

void Edge::destructOwnedHeap()
{
   delete _preNode;
   delete _postNode;
}

std::string Edge::getModuleTypeName() const
{
   return "edge";
}

void Edge::internalGenerateFiles()
{
   assert(strcmp(getName().c_str(), ""));
   generateInstanceBase();
   generateInstance();
   generateCompCategoryBase();
   generateCompCategory();
   generateFactory();
   generateInAttrPSet();
   generateOutAttrPSet();
   generatePSet();
   generatePublisher();
   generateWorkUnitInstance();
   generateWorkUnitShared();
   generateSharedMembers();
   generateTriggerableCallerInstance();
//    generateInstanceProxy();      // commented out by Jizhu Lu on 02/13/2006
}

void Edge::addExtraInstanceBaseMethods(Class& instance) const
{
   SharedCCBase::addExtraInstanceBaseMethods(instance);

   std::string baseName = getType() + "Base";
   std::unique_ptr<BaseClass> base(new BaseClass(baseName));
  
//    CustomAttribute* cusAtt = new CustomAttribute("_edgeCompCategoryBase",
// 						 "EdgeCompCategoryBase");
//    cusAtt->setPointer();
//    cusAtt->setConstructorParameterNameExtra("cc");
//    std::unique_ptr<Attribute> simAtt(cusAtt);
//    base->addAttribute(simAtt);
   instance.addBaseClass(std::move(base));

   instance.addHeader("\"" + baseName + ".h\"");

   // add setPreNode method
   std::unique_ptr<Method> setPreNodeMethod(new Method("setPreNode", "void"));
   setPreNodeMethod->setVirtual();
   setPreNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   std::ostringstream setPreNodeMethodFB;
   if (_preNode) {
      setPreNodeMethodFB
	 << getEdgeConnectionCommonBody(*_preNode);
   }
   setPreNodeMethodFB
      << TAB << "checkAndSetPreNode(" << PREFIX << "node" << ");\n";
   setPreNodeMethod->setFunctionBody(setPreNodeMethodFB.str());
   instance.addMethod(std::move(setPreNodeMethod));

   // add setPostNode method
   std::unique_ptr<Method> setPostNodeMethod(new Method("setPostNode", "void"));
   setPostNodeMethod->setVirtual();
   setPostNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   std::ostringstream setPostNodeMethodFB;
   if (_postNode) {
      setPostNodeMethodFB
	 << getEdgeConnectionCommonBody(*_postNode);
   }
   setPostNodeMethodFB
      << TAB << "checkAndSetPostNode(" << PREFIX << "node" << ");\n";
   setPostNodeMethod->setFunctionBody(setPostNodeMethodFB.str());
   instance.addMethod(std::move(setPostNodeMethod));

   if (_preNode) {
      _preNode->addInterfaceHeaders(instance);
   }
   if (_postNode) {
      _postNode->addInterfaceHeaders(instance);
   }
}

void Edge::addExtraInstanceProxyMethods(Class& instance) const
{
   SharedCCBase::addExtraInstanceProxyMethods(instance);

   std::string baseName = getType() + "ProxyBase";
   std::unique_ptr<BaseClass> base(new BaseClass(baseName));
  
//    CustomAttribute* cusAtt = new CustomAttribute("_edgeCompCategoryBase",
// 						 "EdgeCompCategoryBase");
//    cusAtt->setPointer();
//    cusAtt->setConstructorParameterNameExtra("cc");
//    std::unique_ptr<Attribute> simAtt(cusAtt);
//    base->addAttribute(simAtt);
   instance.addBaseClass(std::move(base));

   instance.addHeader("\"" + baseName + ".h\"");
}

void Edge::addExtraInstanceMethods(Class& instance) const
{
   SharedCCBase::addExtraInstanceMethods(instance);

   std::string baseName = getInstanceBaseName();
   std::unique_ptr<BaseClass> base(
      new BaseClass(baseName));

//    CustomAttribute* cusAtt = new CustomAttribute("_edgeCompCategoryBase",
// 						 "EdgeCompCategoryBase");
//    cusAtt->setPointer();
//    cusAtt->setConstructorParameterNameExtra("cc");
//    std::unique_ptr<Attribute> simAtt(cusAtt);
//    base->addAttribute(simAtt);
   instance.addBaseClass(std::move(base));

   instance.addHeader("\"" + baseName + ".h\"");
   instance.addStandardMethods();
}

void Edge::addExtraCompCategoryBaseMethods(Class& instance) const
{
   SharedCCBase::addExtraCompCategoryBaseMethods(instance);
   instance.addHeader("\"" + getInstanceName() + ".h\"");
//   instance.addHeader("\"ConnectionIncrement.h\"");

   CustomAttribute* edgeListAtt = 
      new CustomAttribute("_edgeList", 
			  "ShallowArray<" + getInstanceName() + 
			  ", 1000, 8>");  
   std::unique_ptr<Attribute> edgeListAttAP(edgeListAtt);
   instance.addAttribute(edgeListAttAP);

   CustomAttribute* computeCostAtt = 
      new CustomAttribute("_computeCost", 
			  "ConnectionIncrement");  
   std::unique_ptr<Attribute> computeCostAttAP(computeCostAtt);
   instance.addAttribute(computeCostAttAP);

   // add getEdge Method
   std::unique_ptr<Method> getEdgeMethod(
      new Method("getEdge", "Edge*") );
   getEdgeMethod->setVirtual(true);
   std::ostringstream getEdgeMethodFunctionBody;
   getEdgeMethodFunctionBody 
       << TAB << "_edgeList.increase();\n"
       << TAB 
       << "_edgeList[_edgeList.size() - 1].setEdgeCompCategoryBase(this);\n"
       << TAB << "return &(_edgeList[_edgeList.size() - 1]);\n";
//       << TAB << "Edge* e = new " << getInstanceName() << "(this);\n"
//       << TAB << "_edgeList.push_back(e);\n"
//       << TAB << "return e;\n";
   getEdgeMethod->setFunctionBody(
      getEdgeMethodFunctionBody.str());
   instance.addMethod(std::move(getEdgeMethod));

   // add getComputeCost method
   std::unique_ptr<Method> getComputeCostMethod(
      new Method("getComputeCost", "ConnectionIncrement*"));
   getComputeCostMethod->setVirtual();
//   getComputeCostMethod->setConst();
   getComputeCostMethod->setFunctionBody(TAB + "return &_computeCost;\n");
   instance.addMethod(std::move(getComputeCostMethod));

   // add getNumOfEdges Method
   std::unique_ptr<Method> getNumOfEdgesMethod(
      new Method("getNumOfEdges", "int") );
   getNumOfEdgesMethod->setVirtual(true);
   std::ostringstream getNumOfEdgesMethodFB;
   getNumOfEdgesMethodFB
      << TAB << "return _edgeList.size();\n";
   getNumOfEdgesMethod->setFunctionBody(
      getNumOfEdgesMethodFB.str());
   instance.addMethod(std::move(getNumOfEdgesMethod));
}

void Edge::addExtraCompCategoryMethods(Class& instance) const
{
   SharedCCBase::addExtraCompCategoryMethods(instance);
   instance.addClass(getInstanceName());
}

std::string Edge::getEdgeConnectionCommonBody(
   EdgeConnection& connection) const
{
   std::ostringstream os;
   std::set<std::string> interfaceCasts = 
      connection.getInterfaceCasts(PREFIX + "node");
   std::set<std::string>::iterator it, end = interfaceCasts.end();
   for (it = interfaceCasts.begin(); it != end; ++it) {
      os << TAB << *it;
   }   
   os << connection.getConnectionCode(getName(), "");
   return os.str();
}

void Edge::addCompCategoryBaseConstructorMethod(Class& instance) const
{
   // Constructor 
   std::unique_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(getCompCategoryBaseName()));
   constructor->addParameter("Simulation& sim");
   constructor->addParameter("const std::string& modelName");
   constructor->addParameter("const NDPairList& ndpList");
   constructor->setInitializationStr(
      getFrameworkCompCategoryName() + "(sim, modelName)");
   constructor->setFunctionBody(getCompCategoryBaseConstructorBody());
   std::unique_ptr<Method> consToIns(constructor.release());
   instance.addMethod(std::move(consToIns));
}
