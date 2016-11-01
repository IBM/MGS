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

#include "Node.h"
#include "SharedCCBase.h"
#include "Generatable.h"
#include "DataType.h"
#include "AccessType.h"
#include "BaseClass.h"
#include "Class.h"
#include "ConstructorMethod.h"
#include "Method.h"
#include "CustomAttribute.h"
#include "Attribute.h"
#include "MacroConditional.h"
#include <memory>
#include <string>
#include <cassert>
#include <iostream>
#include <stdio.h>
#include <string.h>

Node::Node(const std::string& fileName)
   : SharedCCBase(fileName)
{
}


std::string Node::getType() const
{
   return "Node";
}

void Node::duplicate(std::auto_ptr<Generatable>& rv) const
{
   rv.reset(new Node(*this));
}

Node::~Node() 
{
}

std::string Node::getModuleTypeName() const
{
   return "node";
}

void Node::internalGenerateFiles() 
{
   assert(strcmp(getName().c_str(), ""));

   generateInstanceBase();
   generateInstance();
   generateCompCategoryBase();
   generateCompCategory();
   generateFactory();
   generateGridLayerData();
   generateNodeAccessor();
   generateInAttrPSet();
   generateOutAttrPSet();
   generatePSet();
   generatePublisher();
   generateWorkUnitGridLayers();
   generateWorkUnitInstance();
   generateWorkUnitShared();
   generateSharedMembers();
   generateTriggerableCallerInstance();
   generateInstanceProxy();
   generateResourceFile();
}

void Node::generateResourceFile()
{
//   std::cout << "generateResourceFile." << std::endl;
}

void Node::generateGridLayerData()
{
   std::string gridLayerDescriptor = "GridLayerDescriptor";

   MacroConditional mpiConditional(MPICONDITIONAL);

   std::auto_ptr<Class> instance(new Class(getGridLayerDataName()));
   instance->addClass(getRelationalDataUnitName());
   instance->addClass(getCompCategoryBaseName());
   instance->addClass(getInstanceName());
   instance->addClass(gridLayerDescriptor);
   instance->addHeader("\"NodeInstanceAccessor.h\"");
   instance->addHeader("\"GridLayerData.h\"");
   instance->addHeader("\"Grid.h\"");
   instance->addHeader("\"" + getInstanceProxyName() + ".h\"", MPICONDITIONAL);
   instance->addHeader("\"ShallowArray.h\"", MPICONDITIONAL);

   std::auto_ptr<BaseClass> base(new BaseClass("GridLayerData"));
   instance->addBaseClass(base);

   CustomAttribute* nodeInstanceAccessors = new CustomAttribute("_nodeInstanceAccessors", "NodeInstanceAccessor");
   nodeInstanceAccessors->setCArray();
   nodeInstanceAccessors->setOwned();
   nodeInstanceAccessors->setPointer();
   
   std::auto_ptr<Attribute> nodeInstanceAccessorsAp(nodeInstanceAccessors);
   instance->addAttribute(nodeInstanceAccessorsAp);

   // !!! Important the proxy nodes should be put in a container that 
   // doesn't copy its internals, or owned pointers should be used.
   // For example, we couldn't use an std::vector here...

   // Constructor 
   std::auto_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(getGridLayerDataName()));
   constructor->addParameter(getCompCategoryBaseName() + "* compCategory");
   constructor->addParameter(gridLayerDescriptor + "* gridLayerDescriptor");
   constructor->addParameter("int gridLayerIndex");
   constructor->setInitializationStr(
      "GridLayerData(compCategory, gridLayerDescriptor, gridLayerIndex)");
   std::ostringstream constructorFB;
   constructorFB
      << TAB << "_nodeInstanceAccessors = new NodeInstanceAccessor[_nbrUnits];\n"
      << TAB <<"// set gridNode index for each node's relational information\n"
      << TAB << "int top;\n"
      << TAB << "int uniformDensity = _gridLayerDescriptor->isUniform();\n"
      << TAB << "int gridNodes = _gridLayerDescriptor->getGrid()->getNbrGridNodes();\n"
      << TAB << "Simulation *sim = &compCategory->getSimulation();\n"
      << TAB << "unsigned my_rank = sim->getRank();\n"
      << TAB << "for(int n = 0, gn = 0; gn < gridNodes; ++gn) {\n"
      << TAB << TAB << "if (uniformDensity) {\n"
      << TAB << TAB << TAB << "top = (gn + 1) * uniformDensity;\n"
      << TAB << TAB << "} else {\n"
      << TAB << TAB << TAB
      << "top = _nodeOffsets[gn] + _gridLayerDescriptor->getDensity(gn);\n"
      << TAB << TAB << "}\n"
      << TAB << TAB << "for (; n < top; ++n) {\n"
      << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setNodeIndex(gn);\n"
      << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setIndex(n);\n"
      << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setGridLayerData(this);\n\n"
      << TAB << TAB << TAB << "if (sim->isSimulatePass() && sim->getGranule(_nodeInstanceAccessors[n])->getPartitionId() == my_rank) {\n" 
//      << TAB << TAB << TAB << "if (!sim->isDistributed() || sim->getGranule(_nodeInstanceAccessors[n])->getGraphId() == my_rank) {\n"
      << TAB << TAB << TAB << TAB << "compCategory->allocateNode(&_nodeInstanceAccessors[n]);\n"
      << TAB << TAB << TAB << "}\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n";


   constructor->setFunctionBody(constructorFB.str());
   std::auto_ptr<Method> consToIns(constructor.release());
   instance->addMethod(consToIns);

   // add getNodeInstanceAccessors method
   std::auto_ptr<Method> getNodeInstanceAccessorsMethod(new Method("getNodeInstanceAccessors", 
						   "NodeInstanceAccessor*"));
   getNodeInstanceAccessorsMethod->setFunctionBody(
      TAB + "return _nodeInstanceAccessors;\n");
   instance->addMethod(getNodeInstanceAccessorsMethod);

   instance->addBasicDestructor();
   _classes.push_back(instance.release());
}

void Node::addExtraInstanceBaseMethods(Class& instance) const
{
   SharedCCBase::addExtraInstanceBaseMethods(instance);

   instance.setCopyingRemoved();

   std::string baseName = getType() + "Base";
   std::auto_ptr<BaseClass> base(new BaseClass(baseName));
  
   instance.addBaseClass(base);

   // addPostVariable method
   std::auto_ptr<Method> addPostVariableMethod(new Method("addPostVariable", 
							  "void"));
   addPostVariableMethod->setVirtual();
   addPostVariableMethod->addParameter("VariableDescriptor* " + PREFIX + "variable");
   addPostVariableMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostVariableMethod->setFunctionBody(getAddPostVariableFunctionBody());
   instance.addMethod(addPostVariableMethod);

   // addPostEdge method
   std::auto_ptr<Method> addPostEdgeMethod(new Method("addPostEdge", "void"));
   addPostEdgeMethod->setVirtual();
   addPostEdgeMethod->addParameter("Edge* " + PREFIX + "edge");
   addPostEdgeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostEdgeMethod->setFunctionBody(getAddPostEdgeFunctionBody());
   instance.addMethod(addPostEdgeMethod);

   // addPostNode method
   std::auto_ptr<Method> addPostNodeMethod(new Method("addPostNode", "void"));
   addPostNodeMethod->setVirtual();
   addPostNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   addPostNodeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostNodeMethod->setFunctionBody(getAddPostNodeFunctionBody());
   instance.addMethod(addPostNodeMethod);

   // addPreConstant method
   std::auto_ptr<Method> addPreConstantMethod(new Method("addPreConstant", 
							 "void"));
   addPreConstantMethod->setVirtual();
   addPreConstantMethod->addParameter("Constant* " + PREFIX + "constant");
   addPreConstantMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreConstantMethod->setFunctionBody(getAddPreConstantFunctionBody());
   instance.addMethod(addPreConstantMethod);

   // addPreVariable method
   std::auto_ptr<Method> addPreVariableMethod(new Method("addPreVariable", 
							 "void"));
   addPreVariableMethod->setVirtual();
   addPreVariableMethod->addParameter("VariableDescriptor* " + PREFIX + "variable");
   addPreVariableMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreVariableMethod->setFunctionBody(getAddPreVariableFunctionBody());
   instance.addMethod(addPreVariableMethod);

   // addPreEdge method
   std::auto_ptr<Method> addPreEdgeMethod(new Method("addPreEdge", "void"));
   addPreEdgeMethod->setVirtual();
   addPreEdgeMethod->addParameter("Edge* " + PREFIX + "edge");
   addPreEdgeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreEdgeMethod->setFunctionBody(getAddPreEdgeFunctionBody());
   instance.addMethod(addPreEdgeMethod);

   // addPreNode method
   std::auto_ptr<Method> addPreNodeMethod(new Method("addPreNode", "void"));
   addPreNodeMethod->setVirtual();
   addPreNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   addPreNodeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreNodeMethod->setFunctionBody(getAddPreNodeFunctionBody());
   instance.addMethod(addPreNodeMethod);

   instance.addHeader("\"" + baseName + ".h\"");

   // add getComputeCost method
   std::auto_ptr<Method> getComputeCostMethod(
      new Method("getComputeCost", "ConnectionIncrement*"));
   getComputeCostMethod->setVirtual();
   getComputeCostMethod->setConst();
   getComputeCostMethod->setFunctionBody("#if 0\n" + TAB + "return &_computeCost;\n" + "#endif\n" + TAB + "return NULL;\n"); // modified by Jizhu Lu on 04/06/2006 to temporarilly disable computeCost-related
   instance.addMethod(getComputeCostMethod);

}

void Node::addExtraInstanceProxyMethods(Class& instance) const
{
   SharedCCBase::addExtraInstanceProxyMethods(instance);

   instance.setCopyingRemoved();

   std::string baseName = getType() + "ProxyBase";
   std::auto_ptr<BaseClass> base(new BaseClass(baseName));
  
   instance.addBaseClass(base);

   instance.addHeader("\"" + baseName + ".h\"");

   // addPostVariable method
   std::auto_ptr<Method> addPostVariableMethod(new Method("addPostVariable", 
							  "void"));
   addPostVariableMethod->setVirtual();
   addPostVariableMethod->addParameter("VariableDescriptor* " + PREFIX + "variable");
   addPostVariableMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostVariableMethod->setFunctionBody(getAddPostVariableFunctionBody());
   instance.addMethod(addPostVariableMethod);

   // addPostEdge method
   std::auto_ptr<Method> addPostEdgeMethod(new Method("addPostEdge", "void"));
   addPostEdgeMethod->setVirtual();
   addPostEdgeMethod->addParameter("Edge* " + PREFIX + "edge");
   addPostEdgeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostEdgeMethod->setFunctionBody(getAddPostEdgeFunctionBody());
   instance.addMethod(addPostEdgeMethod);

   // addPostNode method
   std::auto_ptr<Method> addPostNodeMethod(new Method("addPostNode", "void"));
   addPostNodeMethod->setVirtual();
   addPostNodeMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   addPostNodeMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPostNodeMethod->setFunctionBody(getAddPostNodeFunctionBody());
   instance.addMethod(addPostNodeMethod);
}

void Node::addExtraInstanceMethods(Class& instance) const
{
   SharedCCBase::addExtraInstanceMethods(instance);

   instance.setCopyingRemoved();

   std::string baseName = getInstanceBaseName();
   std::auto_ptr<BaseClass> base(
      new BaseClass(baseName));
   instance.addBaseClass(base);

   instance.addHeader("\"" + baseName + ".h\"");
   instance.addHeader("\"rndm.h\"");
   instance.addBasicDestructor();
}

void Node::addExtraCompCategoryBaseMethods(Class& instance) const
{
   SharedCCBase::addExtraCompCategoryBaseMethods(instance);
   
   instance.addHeader("\"" + getNodeAccessorName() + ".h\"");
   instance.addHeader("\"" + getGridLayerDataName() + ".h\"");
   instance.addHeader("\"GridLayerData.h\"");
   instance.addHeader("\"" + getWorkUnitGridLayersName() + ".h\"");
   instance.addHeader("\"" + getInstanceName() + ".h\"");
   instance.addHeader("\"GridLayerDescriptor.h\"");

   // Add getNodeAccessor method
   std::auto_ptr<Method> getNodeAccessorMethod(
      new Method("getNodeAccessor", "void"));
   getNodeAccessorMethod->setVirtual();
   getNodeAccessorMethod->addParameter(
      "std::auto_ptr<NodeAccessor>& nodeAccessor");
   getNodeAccessorMethod->addParameter(
      "GridLayerDescriptor* gridLayerDescriptor");
   std::ostringstream getNodeAccessorMethodFB;   
   getNodeAccessorMethodFB 
      << TAB << getGridLayerDataName()
      << "* currentGridLayerData = new " << getGridLayerDataName() 
      << "(this, gridLayerDescriptor, _gridLayerDataArraySize);\n"
      << TAB << "_gridLayerDataList.push_back(currentGridLayerData);\n"
      << TAB << "_gridLayerDataArraySize++;\n"
      << TAB << "_gridLayerDataOffsets.push_back(_gridLayerDataOffsets.back()+currentGridLayerData->getNbrUnits());\n"
      << TAB << "nodeAccessor.reset(new " << getNodeAccessorName() 
      << "(getSimulation(), gridLayerDescriptor, currentGridLayerData));\n";
   getNodeAccessorMethod->setFunctionBody(getNodeAccessorMethodFB.str());
   instance.addMethod(getNodeAccessorMethod);

   // Add allocateNode method
   std::auto_ptr<Method> allocateNodeMethod(
      new Method("allocateNode", "void"));
   allocateNodeMethod->addParameter(
      "NodeDescriptor* nd");
   std::ostringstream allocateNodeMethodFB;   
   allocateNodeMethodFB
      << TAB << "_nodes.increaseSizeTo(_nodes.size()+1);\n"
      << TAB << "_nodes[_nodes.size()-1].setNodeDescriptor(nd);\n"
      << TAB << "nd->setNode(&_nodes[_nodes.size()-1]);\n"
      << TAB << "nd->getGridLayerData()->incrementNbrNodesAllocated();\n";
 
   allocateNodeMethod->setFunctionBody(
      allocateNodeMethodFB.str());
   instance.addMethod(allocateNodeMethod);

   // Add getNbrComputationalUnits method
   std::auto_ptr<Method> getNbrComputationalUnitsMethod(
      new Method("getNbrComputationalUnits", "int"));
   std::ostringstream getNbrComputationalUnitsMethodFB;   
   getNbrComputationalUnitsMethodFB
      << TAB << "return _nodes.size();\n";
   getNbrComputationalUnitsMethod->setFunctionBody(
      getNbrComputationalUnitsMethodFB.str());
   instance.addMethod(getNbrComputationalUnitsMethod);

   // Add getComputeCost method
   std::auto_ptr<Method> getComputeCostMethod(
      new Method("getComputeCost", "ConnectionIncrement*"));
   std::ostringstream getComputeCostMethodFB;   
   getComputeCostMethodFB
      << TAB << "return &_computeCost;\n";
   getComputeCostMethod->setFunctionBody(
      getComputeCostMethodFB.str());
   instance.addMethod(getComputeCostMethod);

   MacroConditional mpiConditional(MPICONDITIONAL);

   // added by Jizhu Lu on 04/26/2006
   CustomAttribute* demarshallerMap = new CustomAttribute("_demarshallerMap", "std::map <int, CCDemarshaller*>");
   std::auto_ptr<Attribute> demarshallerMapAptr(demarshallerMap);
   demarshallerMap->setAccessType(AccessType::PROTECTED);   
   demarshallerMap->setMacroConditional(mpiConditional);
   instance.addAttribute(demarshallerMapAptr);

   CustomAttribute* demarshallerMapIter = new CustomAttribute("_demarshallerMapIter", "std::map <int, CCDemarshaller*>::iterator");
   std::auto_ptr<Attribute> demarshallerMapIterAptr(demarshallerMapIter);
   demarshallerMapIter->setAccessType(AccessType::PROTECTED);   
   demarshallerMapIter->setMacroConditional(mpiConditional);
   instance.addAttribute(demarshallerMapIterAptr);
   /************************************/

   CustomAttribute* sendMap = new CustomAttribute("_sendMap", "std::map <int, ShallowArray<" + getInstanceBaseName() + "*> >");
   std::auto_ptr<Attribute> sendMapAptr(sendMap);
   sendMap->setAccessType(AccessType::PROTECTED);   
   sendMap->setMacroConditional(mpiConditional);
   instance.addAttribute(sendMapAptr);

   CustomAttribute* sendMapIter = new CustomAttribute("_sendMapIter", "std::map <int, ShallowArray<" + getInstanceBaseName() + "*> >::iterator");
   std::auto_ptr<Attribute> sendMapIterAptr(sendMapIter);
   sendMapIter->setAccessType(AccessType::PROTECTED);   
   sendMapIter->setMacroConditional(mpiConditional);
   instance.addAttribute(sendMapIterAptr);

   CustomAttribute* nodes = new CustomAttribute("_nodes", "ShallowArray<" + getInstanceName() + ", 1000, 4>");
   // QUESTION: should this shallow array be setOwned? JK, RR, DL 11/29/05
   std::auto_ptr<Attribute> nodesAp(nodes);

   std::ostringstream nodesDeleteString;
   nodesDeleteString << "#ifdef HAVE_MPI\n"
                     << TAB << "std::map<int, CCDemarshaller*>::iterator end2 = _demarshallerMap.end();\n"
                     << TAB << "for (std::map<int, CCDemarshaller*>::iterator iter2=_demarshallerMap.begin(); iter2!=end2; ++iter2) {\n"
                     << TAB << TAB << "delete (*iter2).second;\n"
                     << TAB << "}\n"
		     << "#endif\n"
		     << TAB << "if (CG_sharedMembers) {\n"
		     << TAB << TAB << "delete CG_sharedMembers;\n"
		     << TAB << TAB << "CG_sharedMembers=0;\n"
		     << TAB << "}\n";

   nodes->setCustomDeleteString(nodesDeleteString.str());
   nodes->setAccessType(AccessType::PROTECTED);
   instance.addAttribute(nodesAp);

   CustomAttribute* compCost = new CustomAttribute("_computeCost", "ConnectionIncrement");
   std::auto_ptr<Attribute> compCostAp(compCost);
   compCost->setAccessType(AccessType::PROTECTED);
   instance.addAttribute(compCostAp);

   // Destructor 
   instance.addBasicDestructor();
}

void Node::addExtraCompCategoryMethods(Class& instance) const
{
   SharedCCBase::addExtraCompCategoryMethods(instance);
}

void Node::generateWorkUnitGridLayers()
{
   generateWorkUnitCommon("GridLayers", "GridLayerData", 
			  getCompCategoryBaseName());
}

void Node::generateNodeAccessor()
{
   std::auto_ptr<Class> instance(new Class(getNodeAccessorName()));
   instance->addClass("Simulation");  
   instance->addClass("Node");  
   instance->addClass("GridLayerDescriptor");  
   instance->addClass(getGridLayerDataName());  
   instance->addHeader("\"NodeAccessor.h\"");
   instance->addHeader("\"" + getInstanceName() + ".h\"");
   instance->addHeader("<memory>");
   instance->addHeader("<vector>");
   instance->addHeader("<string>");
   instance->addExtraSourceHeader("\"SyntaxErrorException.h\"");

   std::auto_ptr<BaseClass> base(new BaseClass("NodeAccessor"));
   instance->addBaseClass(base);

   CustomAttribute* sim = new CustomAttribute("_sim", "Simulation");
   sim->setConstructorParameterNameExtra("sim");   
   sim->setAccessType(AccessType::PRIVATE);   
   sim->setReference();   
   std::auto_ptr<Attribute> simAp(sim);
   instance->addAttribute(simAp);

   CustomAttribute* gldesc = new CustomAttribute("_gridLayerDescriptor",
					      "GridLayerDescriptor");
   gldesc->setPointer();   
   gldesc->setConstructorParameterNameExtra("gridLayerDescriptor");   
   gldesc->setAccessType(AccessType::PRIVATE);   
   std::auto_ptr<Attribute> gldescAp(gldesc);
   instance->addAttribute(gldescAp);

   CustomAttribute* gldata = new CustomAttribute("_gridLayerData",
						 getGridLayerDataName());
   gldata->setPointer();   
   gldata->setConstructorParameterNameExtra("gridLayerData");   
   gldata->setAccessType(AccessType::PRIVATE);   
   std::auto_ptr<Attribute> gldataAp(gldata);
   instance->addAttribute(gldataAp);

   // add getNbrUnits method
   std::auto_ptr<Method> getNbrUnitsMethod(
      new Method("getNbrUnits", 
		 "int"));
   getNbrUnitsMethod->setFunctionBody(
      TAB + "return _gridLayerData->getNbrUnits();\n");
   getNbrUnitsMethod->setVirtual();
   instance->addMethod(getNbrUnitsMethod);

   // add getGridLayerDescriptor method
   std::auto_ptr<Method> getGridLayerDescriptorMethod(
      new Method("getGridLayerDescriptor", 
		 "GridLayerDescriptor*"));
   getGridLayerDescriptorMethod->setFunctionBody(
      TAB + "return _gridLayerDescriptor;\n");
   getGridLayerDescriptorMethod->setVirtual();
   instance->addMethod(getGridLayerDescriptorMethod);

   // add getModelName method
   std::auto_ptr<Method> getModelNameMethod(
      new Method("getModelName", 
		 "std::string"));
   getModelNameMethod->setFunctionBody(
      TAB + "return _gridLayerDescriptor->getModelName();\n");
   getModelNameMethod->setVirtual();
   instance->addMethod(getModelNameMethod);

   // add getNodeDescriptor1 method
   std::auto_ptr<Method> getNodeDescriptor1Method(
      new Method("getNodeDescriptor", "NodeDescriptor*"));
   getNodeDescriptor1Method->addParameter("const std::vector<int>& coords");
   getNodeDescriptor1Method->addParameter("int densityIndex");   
   getNodeDescriptor1Method->setFunctionBody(
      TAB + "return getNodeDescriptor(_gridLayerDescriptor->getGrid()->getNodeIndex(coords), densityIndex);\n");
   getNodeDescriptor1Method->setVirtual();
   instance->addMethod(getNodeDescriptor1Method);

   // add getNodeDescriptor2 method
   std::auto_ptr<Method> getNodeDescriptor2Method(
      new Method("getNodeDescriptor", "NodeDescriptor*"));
   getNodeDescriptor2Method->addParameter("int nodeIndex");
   getNodeDescriptor2Method->addParameter("int densityIndex");   
   std::ostringstream getNodeDescriptor2MethodFB;
   getNodeDescriptor2MethodFB
      << TAB << "int density = _gridLayerDescriptor->getDensity(nodeIndex);\n"
      << TAB << "if (densityIndex >= density) {\n"
      //<< TAB << TAB << "std::cerr << DENSITY_ERROR_MESSAGE << std::endl;\n"
      << TAB << TAB << "std::cerr << \" " 
	  << getNodeAccessorName()  << " \" << DENSITY_ERROR_MESSAGE << std::endl;\n"
      << TAB << TAB << "throw SyntaxErrorException(DENSITY_ERROR_MESSAGE);\n"
      << TAB << "}\n"
      << TAB << "if (_gridLayerDescriptor->isUniform()) {\n"
      << TAB << TAB 
      << "// idx can be computed by simply multiplying by density\n"
      << TAB << TAB << "nodeIndex *= density;\n"
      << TAB << "} else {\n"
      << TAB << TAB 
      << "// density must be non-uniform, use nodeOffsets set by CompCategory\n"
      << TAB << TAB << "if (_gridLayerData->getNodeOffsets().size()) {\n"
      << TAB << TAB << TAB 
      << "nodeIndex = _gridLayerData->getNodeOffsets()[nodeIndex];\n"
      << TAB << TAB << "} else {\n"
      << TAB << TAB << TAB 
      << "std::cerr << OFFSET_ERROR_MESSAGE << std::endl;\n"
      << TAB << TAB << TAB 
      << "throw SyntaxErrorException(OFFSET_ERROR_MESSAGE);\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n"
      << TAB << "nodeIndex += densityIndex;\n"
      << TAB << "return (_gridLayerData->getNodeInstanceAccessors()) + nodeIndex;\n";   
   
   getNodeDescriptor2Method->setFunctionBody(getNodeDescriptor2MethodFB.str());
   getNodeDescriptor2Method->setVirtual();
   instance->addMethod(getNodeDescriptor2Method);

   instance->addStandardMethods();   
   _classes.push_back(instance.release());
}

void Node::addCompCategoryBaseConstructorMethod(Class& instance) const
{
   // Constructor 
   std::auto_ptr<ConstructorMethod> constructor(
      new ConstructorMethod(getCompCategoryBaseName()));
   constructor->addParameter("Simulation& sim");
   constructor->addParameter("const std::string& modelName");
   constructor->addParameter("const NDPairList& ndpList");
   constructor->setInitializationStr(
      getFrameworkCompCategoryName() + 
      "(sim, modelName)");
   constructor->setFunctionBody(getCompCategoryBaseConstructorBody());
   std::auto_ptr<Method> consToIns(constructor.release());
   instance.addMethod(consToIns);
}
