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
   addSupportForMachineType(MachineType::GPU);

   std::unique_ptr<Class> compcat_ptr; 
   {
   auto classType = std::make_pair(Class::PrimeType::Node, Class::SubType::BaseClass);
   bool use_classType = true;
   compcat_ptr = generateInstanceBase(use_classType, classType); //CG_LifeNode.h/,C
   }
   //compcat_ptr = generateInstanceBase(); //CG_LifeNode.h/,C
   auto classType = std::make_pair(Class::PrimeType::Node, Class::SubType::Class);
   bool use_classType = true;
   generateInstance(use_classType, classType, compcat_ptr.get());  //LifeNode.h/C.gen
   generateCompCategoryBase(compcat_ptr.release()); //CG_LifeNodeCompCategory.h/.C
   generateCompCategory(); //LifeNodeCompCategory.h/.C.gen
   generateFactory();
   generateGridLayerData();
   generateNodeAccessor();
   generateInAttrPSet();
   generateOutAttrPSet();
   {
      auto classType = std::make_pair(Class::PrimeType::Node, Class::SubType::BaseClassPSet);
      bool use_classType = true;
      generatePSet(use_classType, classType); //CG_LifeNodePSet.h/.C
   }
   generatePublisher();
   generateWorkUnitGridLayers();
   generateWorkUnitInstance();
   generateWorkUnitShared();
   generateSharedMembers();
   generateTriggerableCallerInstance();
   {
   auto classType = std::make_pair(Class::PrimeType::Node, Class::SubType::BaseClassProxy );
   bool use_classType = true;
   generateInstanceProxy(use_classType, classType);
   }
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

   std::ostringstream customDeleteString;   
   customDeleteString << 
"#ifdef REUSE_NODEACCESSORS\n"
      << TAB << 
      "Simulation *sim = &(_nodeInstanceAccessors[0].getGridLayerData()->getNodeCompCategoryBase()->getSimulation());\n"
      << TAB << 
      "if (sim->isSimulatePass())\n"
      << TAB << TAB << 
      "delete[] _nodeInstanceAccessors;\n"
"#else\n"
      << TAB << 
      "delete[] _nodeInstanceAccessors;\n"
"#endif\n";
   std::string deleteString(customDeleteString.str());
   nodeInstanceAccessors->setCompleteCustomDeleteString(deleteString);
   
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

   std::string className = getInstanceName();
   constructorFB
      << TAB << "Simulation *sim = &compCategory->getSimulation();\n"
      << "#ifdef REUSE_NODEACCESSORS\n"
      << TAB << 
      "if (sim->isGranuleMapperPass())\n"
      << TAB << 
      "{\n"
      << TAB << TAB << 
       "if (sim->nodeInstanceAccessor.count(\"" << className << "\") == 0)\n"
      << TAB << TAB << 
       "{\n"
      << TAB << TAB << TAB << 
           "std::map<int, NodeInstanceAccessor*> gridlayer_2_NA;\n"
      << TAB << TAB << TAB << 
           "sim->nodeInstanceAccessor[\"" << className << "\"] = gridlayer_2_NA;\n"
      << TAB << TAB << 
       "}\n"
      << TAB << TAB << 
       "_nodeInstanceAccessors = new NodeInstanceAccessor[_nbrUnits];\n"
      << TAB << TAB << 
       "sim->nodeInstanceAccessor[\"" << className << "\"][getGridLayerIndex()] =  _nodeInstanceAccessors;\n"
      << TAB << 
      "}\n"
      << TAB << 
      "else{\n"
      << TAB << TAB << 
       "_nodeInstanceAccessors = sim->nodeInstanceAccessor[\"" << className << "\"][getGridLayerIndex()];\n"
      << TAB << 
      "}\n"
      << TAB << 
"#if defined(REUSE_NODEACCESSORS) and defined(TRACK_SUBARRAY_SIZE)\n"
      << TAB << 
      "if (sim->_nodeShared.count(\"" << className << "\") == 0)\n"
      << TAB << 
      "{\n"
      << TAB << TAB << 
	 className << "* ptr = (new " << className << "());\n"
      << TAB << TAB << 
	 "sim->_nodeShared[\"" << className << "\"] = ptr;\n"
      << TAB << 
      "}\n"
      << TAB << 
      "for (int ii = 0; ii < _nbrUnits; ii++)\n"
      << TAB << 
      "    _nodeInstanceAccessors[ii].setSharedNode(sim->_nodeShared[\"" << className << "\"]);\n"
      << TAB << 
"#endif\n"
      <<
"#else\n"
      << TAB << "_nodeInstanceAccessors = new NodeInstanceAccessor[_nbrUnits];\n"
      <<
"#endif\n"
      << TAB <<"// set gridNode index for each node's relational information\n"
      << TAB << "int top;\n"
      << TAB << "int uniformDensity = _gridLayerDescriptor->isUniform();\n"
      << TAB << "int gridNodes = _gridLayerDescriptor->getGrid()->getNbrGridNodes();\n"
      << TAB << "unsigned my_rank = sim->getRank();\n"
      << "#if defined(HAVE_GPU) && defined(__NVCC__)\n"
      //<< TAB << "/*"
      << TAB << "if (sim->isGranuleMapperPass()) {\n"
      //<< TAB << "if (sim->isGranuleMapperPass() || sim->isCostAggregationPass()) {\n"
      << TAB << TAB << "if (sim->_nodes_count.count(\"" << getInstanceName() + "\") == 0)\n"
      << TAB << TAB << "{ \n"
      << TAB << TAB << TAB << "std::vector<int> nodes_on_ranks(sim->getNumProcesses(), 0);\n"
      << TAB << TAB << TAB << "sim->_nodes_count[\"" << getInstanceName() << "\"] = nodes_on_ranks;\n"
      << TAB << "#if defined(DETECT_NODE_COUNT) && DETECT_NODE_COUNT == NEW_APPROACH\n"
      << TAB << TAB << TAB << "std::map<Granule*, int> nodes_on_partitions;\n"
      << TAB << TAB << TAB << "sim->_nodes_granules[\"" << getInstanceName() << "\"] = nodes_on_partitions;\n"
      << TAB << "#else\n"
      //<< TAB << TAB << TAB << "sim->_nodes_count[\"" << getInstanceName() << "\"][my_rank] = 0;\n "
      << TAB << TAB << TAB << "std::vector<Granule*> nodes_on_partitions;\n"
      << TAB << TAB << TAB << "sim->_nodes_granules[\"" << getInstanceName() << "\"] = nodes_on_partitions;\n"
      << TAB << "#endif\n"
      << TAB << TAB << "}\n"
      << TAB << TAB << "for(int n = 0, gn = 0; gn < gridNodes; ++gn) {\n"
      << TAB << TAB << TAB << "if (uniformDensity) {\n"
      << TAB << TAB << TAB << TAB << "top = (gn + 1) * uniformDensity;\n"
      << TAB << TAB << TAB << "} else {\n"
      << TAB << TAB << TAB << TAB
      << TAB << "top = _nodeOffsets[gn] + _gridLayerDescriptor->getDensity(gn);\n"
      << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << "for (; n < top; ++n) {\n"
      << TAB << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setNodeIndex(gn);\n"
      << TAB << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setIndex(n);\n"
      << TAB << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setGridLayerData(this);\n"
      //<< TAB << TAB << TAB << TAB << "if (sim->getGranule(_nodeInstanceAccessors[n])->getPartitionId() == my_rank) {\n" 
      //<< TAB << TAB << TAB << TAB << "if (!sim->isDistributed() || sim->getGranule(_nodeInstanceAccessors[n])->getGraphId() == my_rank) {\n"
      //<< TAB << TAB << TAB << TAB << TAB << "sim->_nodes_count[\"" << getInstanceName() << "\"][my_rank] += 1;;\n"
      //<< TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << TAB <<  "/* it means instance 'LifeNode' at index 'i'\n"
      << TAB << TAB << TAB << TAB <<  " * is created on partition 'sim->_nodes_granules[\"LifeNode\"][i]->getPartitionId()'\n"
      << TAB << TAB << TAB << TAB <<  " */\n"
      << TAB << "#if defined(DETECT_NODE_COUNT) && DETECT_NODE_COUNT == NEW_APPROACH\n"
      << TAB << TAB << TAB << "sim->_nodes_granules[\"" << getInstanceName() << "\"][sim->getGranule(_nodeInstanceAccessors[n])] += 1;\n"
      << TAB << "#else\n"
      << TAB << TAB << TAB << TAB <<  "sim->_nodes_granules[\"" << getInstanceName() << "\"].push_back(sim->getGranule(_nodeInstanceAccessors[n]));\n"
      << TAB << "#endif\n"
      << TAB << TAB << TAB << "}\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n"
      << TAB << "if (sim->isSimulatePass())\n"
      << TAB << "{\n"
      //<< TAB << "    compCategory->allocateNodes(sim->_nodes_count[\"" << getInstanceName() << "\"][my_rank]);\n"
      << TAB << TAB << "/* at the first layer of 'LifeNode' then allocate memory */\n"
      << TAB << TAB << "if (sim->_nodes_count[\"" << getInstanceName() << "\"][my_rank] == 0)\n"
      << TAB << TAB << "{\n"
      << TAB << "#if defined(DETECT_NODE_COUNT) && DETECT_NODE_COUNT == NEW_APPROACH\n"
      << TAB << TAB << TAB << "auto& mymap = sim->_nodes_granules[\"" << getInstanceName() << "\"];\n"
      << TAB << TAB << TAB << "for (auto it = mymap.begin(); it != mymap.end(); ++it)\n"
      << TAB << TAB << TAB << "{\n"
      << TAB << TAB << TAB << TAB << "auto& key = it->first;\n"
      << TAB << TAB << TAB << TAB << "auto& value = it->second;\n"
      << TAB << TAB << TAB << TAB << "if (key->getPartitionId()  == my_rank)\n"
      << TAB << TAB << TAB << TAB << "{\n"
      << TAB << TAB << TAB << TAB << "     sim->_nodes_count[\"" << getInstanceName() << "\"][my_rank] += value;\n"
      << TAB << TAB << TAB << TAB << "}\n"
      << TAB << TAB << TAB << "}\n"
      << TAB << "#else\n"
      //<< TAB << TAB << "   auto tmpNodeAccessor = new NodeInstanceAccessor();\n"
      << TAB << TAB << "   for(int n = 0; n < sim->_nodes_granules[\"" << getInstanceName() << "\"].size(); ++n) {\n"
      << TAB << TAB << "      if (sim->_nodes_granules[\"" << getInstanceName() << "\"][n]->getPartitionId()  == my_rank)\n"
      << TAB << TAB << "      {\n"
      << TAB << TAB << "         sim->_nodes_count[\"" << getInstanceName() << "\"][my_rank] += 1;\n"
      << TAB << TAB << "      }\n"
      << TAB << TAB << "   }\n"
      << TAB << "#endif\n"
      //<< TAB << TAB << "   delete tmpNodeAccessor;\n"
      << TAB << TAB << "   sim->_nodes_granules.erase(\"" << getInstanceName() << "\");\n"
      << TAB << TAB << "   compCategory->allocateNodes(sim->_nodes_count[\"" << getInstanceName() << "\"][my_rank]);\n"
      << TAB << TAB << "}\n"
      << TAB << TAB << "/* at the first layer then allocate all proxies, i.e. CG_LifeNodeProxy\n"
      << TAB << TAB << " */\n"
      << TAB << TAB << "if (sim->_proxy_count.count(\"" << getInstanceName() << "\") == 0)\n"
      << TAB << TAB << "{\n"
      //<< TAB << TAB << "   std::vector<int> proxy_from_ranks(sim->getNumProcesses(), 0);\n"
      << TAB << TAB << "   std::vector<size_t> proxy_from_ranks(sim->getNumProcesses(), 0);\n"
      << TAB << TAB << "   sim->_proxy_count[\"" << getInstanceName() << "\"] = proxy_from_ranks;\n"
      << TAB //<< TAB << "   //std::map<Granule*, std::map<std::string, int>> _granulesFrom_NT_count;
      << TAB << TAB << "   for (auto& kv : sim->_granulesFrom_NT_count)\n"
      << TAB << TAB << "   {\n"
      << TAB << TAB << "      int other_rank = kv.first->getPartitionId();\n"
      << TAB << TAB << "      if (other_rank != my_rank)\n"
      << TAB << TAB << "      {\n"
      << TAB << TAB << "         if (kv.second.count(\"" << getInstanceName() << "\") > 0)\n"
      << TAB << TAB << "            sim->_proxy_count[\"" << getInstanceName() << "\"][other_rank] = kv.second[\"" << getInstanceName() << "\"];\n"
      << TAB << TAB << "      }\n"
      << TAB << TAB << "   }\n"
      << TAB << TAB << "   compCategory->allocateProxies(sim->_proxy_count[\"" << getInstanceName() << "\"]);\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n"         
      //<< TAB << "*/\n"
      << "#endif\n"
      << STR_GPU_CHECK_START
      << "if (! sim->isGranuleMapperPass()) {\n"
      << STR_GPU_CHECK_END
      << TAB << "for(int n = 0, gn = 0; gn < gridNodes; ++gn) {\n"
      << TAB << TAB << "if (uniformDensity) {\n"
      << TAB << TAB << TAB << "top = (gn + 1) * uniformDensity;\n"
      << TAB << TAB << "} else {\n"
      << TAB << TAB << TAB
      << "top = _nodeOffsets[gn] + _gridLayerDescriptor->getDensity(gn);\n"
      << TAB << TAB << "}\n"
      << TAB << TAB << "for (; n < top; ++n) {\n"
      << TAB << 
	 "#if not defined(REUSE_NODEACCESSORS)\n"
      << TAB << 
        "//as this is already set during GRANULE_MAPPER_PASS (see above)\n"
      << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setNodeIndex(gn);\n"
      << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setIndex(n);\n"
      << TAB << 
	 "#endif\n"
      << TAB << TAB << TAB << "_nodeInstanceAccessors[n].setGridLayerData(this);\n\n"
      << TAB << TAB << TAB << "if (sim->isSimulatePass() && sim->getGranule(_nodeInstanceAccessors[n])->getPartitionId() == my_rank) {\n" 
//      << TAB << TAB << TAB << "if (!sim->isDistributed() || sim->getGranule(_nodeInstanceAccessors[n])->getGraphId() == my_rank) {\n"
      << TAB << TAB << TAB << TAB << "compCategory->allocateNode(&_nodeInstanceAccessors[n]);\n"
      << TAB << TAB << TAB << "}\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n"
      << STR_GPU_CHECK_START
      << "}\n"
      << STR_GPU_CHECK_END;

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

   // addPreNode_Dummy method
   //virtual void addPreNode_Dummy(NodeDescriptor* CG_node /*nd_for_the_incoming_node*/, ParameterSet* CG_pset, Simulation* sim, NodeDescriptor* nd_for_this_node);
   std::auto_ptr<Method> addPreNode_DummyMethod(new Method("addPreNode_Dummy", "void"));
   addPreNode_DummyMethod->setVirtual();
   addPreNode_DummyMethod->addParameter("NodeDescriptor* " + PREFIX + "node");
   addPreNode_DummyMethod->addParameter("ParameterSet* " + PREFIX + "pset");
   addPreNode_DummyMethod->addParameter("Simulation* sim");
   addPreNode_DummyMethod->addParameter("NodeDescriptor* nd_for_this_node");
   addPreNode_DummyMethod->setFunctionBody(getAddPreNode_DummyFunctionBody());

   std::vector<std::string> conds;
   conds.push_back(REUSENA_CONDITIONAL);
   conds.push_back(TRACK_SAS_CONDITIONAL);
   MacroConditional trackConnectionConditional(conds);
   addPreNode_DummyMethod->setMacroConditional(trackConnectionConditional);
   instance.addMethod(addPreNode_DummyMethod);
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

void Node::_add_allocateNode_Method(Class& instance) const
{
   std::auto_ptr<Method> allocateNodeMethod(
      new Method("allocateNode", "void"));
   allocateNodeMethod->addParameter(
      "NodeDescriptor* nd");
   std::ostringstream allocateNodeMethodFB;   
   std::vector<std::string> conds;
   conds.push_back(REUSENA_CONDITIONAL);
   conds.push_back(TRACK_SAS_CONDITIONAL);
   MacroConditional trackConnectionConditional(conds);

   std::ostringstream subMethodNoTrackConnection;   
   std::ostringstream subMethodTrackConnection;   

   std::string className = getName();
   subMethodTrackConnection 
      << TAB << "int sz = _nodes.size();\n"
      << TAB << "_nodes[sz-1].setCompCategory(sz-1, this);\n";
      for (auto it = getInstances().begin(); it != getInstances().end(); ++it) {
	 if (it->second->isArray())
	 {
	    std::string tmpVarName = PREFIX_MEMBERNAME + it->first;
	    //NOTE: um_neighbors is an array of array
	    subMethodTrackConnection 
	       << TAB << "{\n"
	       << TAB << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	       << TAB << TAB << tmpVarName << ".increaseSizeTo(sz);\n"
	       << TAB << TAB << "/* find the subarray size for this node */\n"
	       << TAB << TAB <<
	       "int gridLayerIndex = nd->getGridLayerData()->getGridLayerIndex();\n" 
	       << TAB << TAB <<
	       "int nodeAccessor_index = ((NodeInstanceAccessor*)nd)->getIndex();\n" 
	       << TAB << TAB <<
	       "std::pair<int, int> pair_data = std::make_pair(gridLayerIndex, nodeAccessor_index);\n" 
	       << TAB << TAB <<
	       "int subarray_size = _sim._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][pair_data];\n"
	       << TAB << TAB <<
	       PREFIX_MEMBERNAME << it->first << "[sz-1].resize_allocated_subarray(subarray_size, Array_Flat<int>::MemLocation::UNIFIED_MEM);\n"
	       << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
	       << TAB << TAB << tmpVarName << "_start_offset.increaseSizeTo(sz);\n"
	       << TAB << TAB << tmpVarName << "_num_elements.increaseSizeTo(sz);\n"
	       << TAB << TAB << "if (USE_SHARED_MAX_SUBARRAY)\n"
	       << TAB << TAB << "{\n"
	       << TAB << TAB << "//KEEP USING MAX_SUBARRAY_SIZE style, i.e. all nodes share the same MAX_SUBARRAY_SIZE for a specific subarray data member\n"
	       << TAB << TAB << TAB << tmpVarName << "_start_offset[sz-1] = " << tmpVarName << ".size();\n"
	       << TAB << TAB << TAB << tmpVarName << ".increaseSizeTo(sz*getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][std::make_pair(-1,-1)]);\n"
	       << TAB << TAB << "}else{\n"
	       << TAB << TAB << TAB << "// OR using exact-size of subarray for each node\n"
	       << TAB << TAB << TAB << tmpVarName << "_start_offset[sz-1] = " << tmpVarName << ".size();\n"
	       << TAB << TAB << TAB << "int gridLayerIndex = nd->getGridLayerData()->getGridLayerIndex();\n"
	       << TAB << TAB << TAB << "int nodeAccessor_index = ((NodeInstanceAccessor*)nd)->getIndex();\n"
	       << TAB << TAB << TAB << "std::pair<int, int> pair_data = std::make_pair(gridLayerIndex, nodeAccessor_index);\n"
	       << TAB << TAB << TAB << "int subarray_size = _sim._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][pair_data];\n"
	       << TAB << TAB << TAB << tmpVarName << ".increaseSizeTo(" << tmpVarName << ".size() + subarray_size);\n"
	       << TAB << TAB << "}\n"
	       << TAB << TAB << tmpVarName << "_num_elements[sz-1] = 0;\n"
	       << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".increaseSizeTo(sz*" << PREFIX_MEMBERNAME << it->first << "_max_elements);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.increaseSizeTo(sz);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements[sz-1] = 0;\n"
	       << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	       << TAB << TAB << "assert(0);\n"
	       //<< TAB << TAB << "// using exact-size of subarray for each node\n"
	       //<< TAB << TAB << tmpVarName << "_start_offset[sz-1] = " << tmpVarName << ".size();\n"
	       //<< TAB << TAB << "int gridLayerIndex = nd->getGridLayerData()->getGridLayerIndex();\n"
	       //<< TAB << TAB << "int nodeAccessor_index = ((NodeInstanceAccessor*)nd)->getIndex();\n"
	       //<< TAB << TAB << "std::pair<int, int> pair_data = std::make_pair(gridLayerIndex, nodeAccessor_index);\n"
	       //<< TAB << TAB << "int subarray_size = _sim._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][pair_data];\n"
	       //<< TAB << TAB << tmpVarName << ".increaseSizeTo(" << tmpVarName << ".size() + subarray_size);\n"
	       << TAB << "#endif\n"
	       << TAB << "}\n"
	       ;
	 }
	 else{
	    subMethodTrackConnection 
	       << TAB <<PREFIX_MEMBERNAME<< it->first <<  ".increaseSizeTo(sz);\n";
	 }
      }

   subMethodNoTrackConnection 
      << TAB << "int sz = _nodes.size();\n"
      //<< TAB << "_nodes[_nodes.size()-1].setCompCategory(_nodes.size()-1, this);\n";
      << TAB << "_nodes[sz-1].setCompCategory(sz-1, this);\n";
      for (auto it = getInstances().begin(); it != getInstances().end(); ++it) {
	 if (it->second->isArray())
	 {
	    //NOTE: um_neighbors is an array of array
	    subMethodNoTrackConnection 
	       << TAB << "{\n"
	       << TAB << TAB << "int MAX_SUBARRAY_SIZE = " << COMMON_MAX_SUBARRAY_SIZE << ";\n"
	       << TAB << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".increaseSizeTo(sz);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "[sz-1].resize_allocated_subarray(MAX_SUBARRAY_SIZE, " << MEMORY_LOCATION << ");\n"
	       << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".increaseSizeTo(sz*MAX_SUBARRAY_SIZE);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_start_offset.increaseSizeTo(sz);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.increaseSizeTo(sz);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_start_offset[sz-1] = (sz-1) * MAX_SUBARRAY_SIZE;\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements[sz-1] = 0;\n"
	       << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".increaseSizeTo(sz*" << PREFIX_MEMBERNAME << it->first << "_max_elements);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.increaseSizeTo(sz);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements[sz-1] = 0;\n"
	       << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".increaseSizeTo(sz * MAX_SUBARRAY_SIZE);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_start_offset.increaseSizeTo(sz);\n"
	       << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_start_offset[sz-1] = (sz-1) * MAX_SUBARRAY_SIZE;\n"
	       << TAB << "#endif\n"
	       << TAB << "}\n"
	       ;
	 }
	 else{
	    subMethodNoTrackConnection
	       << TAB <<PREFIX_MEMBERNAME<< it->first <<  ".increaseSizeTo(sz);\n";
	 }
      }

   allocateNodeMethodFB
      << TAB << "_nodes.increaseSizeTo(_nodes.size()+1);\n"
      << STR_GPU_CHECK_START
      << trackConnectionConditional.getBeginning()
      << subMethodTrackConnection.str()
      << "#else\n"
      << subMethodNoTrackConnection.str()
      << trackConnectionConditional.getEnding();
   allocateNodeMethodFB  << STR_GPU_CHECK_END
      << TAB << "_nodes[_nodes.size()-1].setNodeDescriptor(nd);\n"
      << TAB << "nd->setNode(&_nodes[_nodes.size()-1]);\n"
      << TAB << "nd->getGridLayerData()->incrementNbrNodesAllocated();\n";
 
   allocateNodeMethod->setFunctionBody(
      allocateNodeMethodFB.str());
   instance.addMethod(allocateNodeMethod);

}

void Node::_add_allocateNodes_Method(Class& instance) const
{
   std::auto_ptr<Method> allocateNodesMethod(
      new Method("allocateNodes", "void"));
   allocateNodesMethod->addParameter(
      "size_t size");
   std::ostringstream allocateNodesMethodFB;   
   std::vector<std::string> conds;
   conds.push_back(REUSENA_CONDITIONAL);
   conds.push_back(TRACK_SAS_CONDITIONAL);
   MacroConditional trackConnectionConditional(conds);

   std::ostringstream subMethodNoTrackConnection;   
   std::ostringstream subMethodTrackConnection;   

   std::string className = getName();

   subMethodTrackConnection << "";
   for (auto it = getInstances().begin(); it != getInstances().end(); ++it) {
      if (it->second->isArray())
      {
	 std::string tmpVarName = PREFIX_MEMBERNAME + it->first;
          //NOTE: um_neighbors is an array of array
         subMethodTrackConnection 
            << TAB << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
            << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n"
            << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
            << TAB << TAB << "size_t MAX_SUBARRAY_SIZE = 0;\n"
            << TAB << TAB << "size_t count = 0;\n"
	    << TAB << TAB << "auto iter = getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"].begin();\n"
	    << TAB << TAB << "auto iend = getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"].end();\n"
      	    << TAB << TAB << "for (; iter < iend; iter++)\n"
      	    << TAB << TAB << "{\n"
      	    << TAB << TAB << "    MAX_SUBARRAY_SIZE = max(MAX_SUBARRAY_SIZE, iter->second);\n"
      	    << TAB << TAB << "    count += iter->second;\n"
      	    << TAB << TAB << "}\n"
	    << TAB << TAB << "std::pair<int, int> key4_max_of_max = std::make_pair(-1,-1);\n"
      	    << TAB << TAB << "getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][key4_max_of_max] = MAX_SUBARRAY_SIZE;\n"
      	    << TAB << TAB << "std::pair<int, int> key4_total = std::make_pair(-2,-2);\n"
      	    << TAB << TAB << "getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][key4_total] = count;\n"
	    << TAB << TAB << "if (USE_SHARED_MAX_SUBARRAY)\n"
      	    << TAB << TAB << "{\n"
      	    << TAB << TAB << "   //with some wasteful - but is based on max-size of that array-datamember\n"
      	    << TAB << TAB << "    um_inputs.resize_allocated(size*MAX_SUBARRAY_SIZE, force_resize);\n"
      	    << TAB << TAB << "}\n"
      	    << TAB << TAB << "else{\n"
      	    << TAB << TAB << "   //exact size needed (behave like OPTION_5)\n"
      	    << TAB << TAB << "    um_inputs.resize_allocated(count, force_resize);\n"
      	    << TAB << TAB << "}\n"
            << TAB << TAB << tmpVarName << "_start_offset.resize_allocated(size, force_resize);\n"
            << TAB << TAB << tmpVarName << "_num_elements.resize_allocated(size, force_resize);\n"
            << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
            << TAB << TAB << "{\n"
            << TAB << TAB << "size_t MAX_SUBARRAY_SIZE = 0;\n"
            << TAB << TAB << "size_t count = 0;\n"
	    << TAB << TAB << "auto iter = getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"].begin();\n"
	    << TAB << TAB << "auto iend = getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"].end();\n"
      	    << TAB << TAB << "for (; iter < iend; iter++)\n"
      	    << TAB << TAB << "{\n"
      	    << TAB << TAB << "    MAX_SUBARRAY_SIZE = max(MAX_SUBARRAY_SIZE, iter->second);\n"
      	    << TAB << TAB << "    count += iter->second;\n"
      	    << TAB << TAB << "}\n"
            << TAB << TAB << tmpVarName << "_max_elements = MAX_SUBARRAY_SIZE;\n"
	    << TAB << TAB << "std::pair<int, int> key4_max_of_max = std::make_pair(-1,-1);\n"
      	    << TAB << TAB << "getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][key4_max_of_max] = MAX_SUBARRAY_SIZE;\n"
      	    << TAB << TAB << "std::pair<int, int> key4_total = std::make_pair(-2,-2);\n"
      	    << TAB << TAB << "getSimulation()._nodes_subarray[\"" << className << "\"][\"" << tmpVarName << "\"][key4_total] = count;\n"
      	    << TAB << TAB << "um_inputs.resize_allocated(size*MAX_SUBARRAY_SIZE, force_resize);\n"
            << TAB << TAB << tmpVarName << "_num_elements.resize_allocated(size, force_resize);\n"
            << TAB << TAB << "}\n"
            << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
            << TAB << TAB << "assert(0);\n"
            << TAB << "#endif\n";
      }
      else{
         subMethodTrackConnection 
            << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n";
      }
   }

   subMethodNoTrackConnection << "";
   for (auto it = getInstances().begin(); it != getInstances().end(); ++it) {
      if (it->second->isArray())
      {
          //NOTE: um_neighbors is an array of array
	 subMethodNoTrackConnection 
	    << TAB << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	    << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n"
	    << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
	    << TAB << TAB << "int MAX_SUBARRAY_SIZE = " << COMMON_MAX_SUBARRAY_SIZE << ";\n"
	    << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size*MAX_SUBARRAY_SIZE, force_resize);\n"
	    << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_start_offset.resize_allocated(size, force_resize);\n"
	    << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.resize_allocated(size, force_resize);\n"
	    << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
	    << TAB << TAB << "{\n"
	    << TAB << TAB << "int MAX_SUBARRAY_SIZE = " << COMMON_MAX_SUBARRAY_SIZE << ";\n"
	    << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_max_elements = MAX_SUBARRAY_SIZE;\n"
	    << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size*" << PREFIX_MEMBERNAME << it->first <<  "_max_elements, force_resize);\n"
	    << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.resize_allocated(size, force_resize);\n"
	    << TAB << TAB << "}\n"
	    << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	    << TAB << TAB << "assert(0);\n"
	    << TAB << "#endif\n";
      }
      else{
	 subMethodNoTrackConnection 
	    << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n";
      }
   }

   allocateNodesMethodFB
      << STR_GPU_CHECK_START
      << TAB << "bool force_resize = true;\n"
      << TAB << "_nodes.resize_allocated(size, force_resize);\n"
      << trackConnectionConditional.getBeginning()
      << subMethodTrackConnection.str()
      << "#else\n"
      << subMethodNoTrackConnection.str()
      << trackConnectionConditional.getEnding();
   allocateNodesMethodFB << STR_GPU_CHECK_END;
   allocateNodesMethod->setFunctionBody(
      allocateNodesMethodFB.str());
   instance.addMethod(allocateNodesMethod);

}

void Node::_add_allocateProxies_Method(Class& instance) const
{
   std::auto_ptr<Method> allocateProxiesMethod(
      new Method("allocateProxies", "void"));
   allocateProxiesMethod->addParameter(
      "const std::vector<size_t>& sizes");
   std::ostringstream allocateProxiesMethodFB;   
   allocateProxiesMethodFB
      << STR_GPU_CHECK_START
      << TAB << "unsigned my_rank = _sim.getRank();\n"
      << TAB << "bool force_resize = true;\n"
      << "#if PROXY_ALLOCATION == OPTION_3\n"
      << TAB << "for (int i = 0; i < _sim.getNumProcesses(); i++)\n"
      << TAB << "{\n"
      << TAB << TAB << "if (i != my_rank)\n"
      << TAB << TAB << "{\n"
      << TAB << TAB << TAB << "CCDemarshaller* ccd = findDemarshaller(i);\n"
      << TAB << TAB << TAB << "int size = sizes[i];\n"
      << TAB << TAB << TAB << "if (size > 0)\n"
      << TAB << TAB << TAB << "{\n"
      << TAB << TAB << TAB << TAB << "ccd->_receiveList.resize_allocated(size, force_resize);\n";
   for (auto it = getInstances().begin(); it != getInstances().end(); ++it) {
      if (it->second->isArray())
      {
	 allocateProxiesMethodFB 
	    << TAB << TAB << TAB << TAB << "ccd->" << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n";
	 //NOTE: um_neighbors is an array of array
	 //allocateProxiesMethodFB 
	 //   << TAB << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n"
	 //   << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size*MAX_SUBARRAY_SIZE, force_resize);\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_start_offset.resize_allocated(size, force_resize);\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.resize_allocated(size, force_resize);\n"
	 //   << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
	 //   << TAB << TAB << "int MAX_SUBARRAY_SIZE = 20;\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_max_elements = MAX_SUBARRAY_SIZE;\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size*um_neighbors_max_elements, force_resize);\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.resize_allocated(size, force_resize);\n"
	 //   << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	 //   << TAB << TAB << "assert(0);\n"
	 //   << TAB << "#endif\n";
      }
      else{
	 allocateProxiesMethodFB 
	    << TAB << TAB << TAB << TAB << "ccd->" << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n";
      }
   }
   allocateProxiesMethodFB 
      << TAB << TAB << TAB << "}\n"
      << TAB << TAB << "}\n"
      << TAB << "}\n";
   allocateProxiesMethodFB
      << "#elif PROXY_ALLOCATION == OPTION_4\n"
   << TAB << "int total = std::accumulate(sizes.begin(), sizes.end(), 0);\n"
   //<< TAB << "assert(0);\n"
   << TAB << "int offset = 0;\n"
   << TAB << "for (int i = 0; i < _sim.getNumProcesses(); i++)\n"
   << TAB << "{\n"
   << TAB << TAB << "offset += sizes[i];\n"
   << TAB << TAB << "if (i != my_rank)\n"
   << TAB << TAB << "{\n"
   << TAB << TAB << TAB << "CCDemarshaller* ccd = findDemarshaller(i);\n"
   << TAB << TAB << TAB << "int size = sizes[i];\n"
   << TAB << TAB << TAB << "if (size > 0)\n"     
   << TAB << TAB << TAB << "{\n"               
   << TAB << TAB << TAB << TAB << "ccd->_receiveList.resize_allocated(size, force_resize);\n"
   << TAB << TAB << TAB << TAB << "ccd->offset = offset;\n"
   << TAB << TAB << TAB << "}\n"                         
   << TAB << TAB << "}\n"                                               
   << TAB << "}\n"                                                 ;
   for (auto it = getInstances().begin(); it != getInstances().end(); ++it) {
      if (it->second->isArray())
      {
	 allocateProxiesMethodFB 
	    << TAB << "this->" << PREFIX_PROXY_MEMBERNAME << it->first << ".resize_allocated(total, force_resize);\n";
         //NOTE: um_neighbors is an array of array
	 //allocateProxiesMethodFB 
	 //   << TAB << "#if DATAMEMBER_ARRAY_ALLOCATION == OPTION_3\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size, force_resize);\n"
	 //   << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size*MAX_SUBARRAY_SIZE, force_resize);\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_start_offset.resize_allocated(size, force_resize);\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.resize_allocated(size, force_resize);\n"
	 //   << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_4b\n"
	 //   << TAB << TAB << "int MAX_SUBARRAY_SIZE = 20;\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_max_elements = MAX_SUBARRAY_SIZE;\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << ".resize_allocated(size*um_neighbors_max_elements, force_resize);\n"
	 //   << TAB << TAB << PREFIX_MEMBERNAME << it->first << "_num_elements.resize_allocated(size, force_resize);\n"
	 //   << TAB << "#elif DATAMEMBER_ARRAY_ALLOCATION == OPTION_5\n"
	 //   << TAB << TAB << "assert(0);\n"
	 //   << TAB << "#endif\n";
      }
      else{
	 allocateProxiesMethodFB 
	    << TAB << "this->" << PREFIX_PROXY_MEMBERNAME << it->first << ".resize_allocated(total, force_resize);\n";
      }
   }
   allocateProxiesMethodFB
      << "#endif //PROXY_ALLOCATION\n";
   //allocateProxiesMethodFB << "#else\n"
   //   << TAB << "_nodes.resize_allocated(size);\n"
   //   << STR_GPU_CHECK_END;
   allocateProxiesMethodFB << STR_GPU_CHECK_END;
   allocateProxiesMethod->setFunctionBody(
      allocateProxiesMethodFB.str());
   instance.addMethod(allocateProxiesMethod);
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
      "std::unique_ptr<NodeAccessor>& nodeAccessor");
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
   this->_add_allocateNode_Method(instance);

   // Add allocateNodes method (i.e. pre-allocate in memory)
   this->_add_allocateNodes_Method(instance);


   // Add allocateProxies method (i.e. pre-allocate in memory)
   this->_add_allocateProxies_Method(instance);

   // Add getNbrComputationalUnits method
   //this->_add_getNbrComputationalUnits_Method(instance);
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

   /*
#if defined(HAVE_GPU) && defined(__NVCC__)
      //ShallowArray<int, Array::MemLocation::UNIFIED_MEM> um_value;
      //ShallowArray<int, Array::MemLocation::UNIFIED_MEM> um_publicValue;
      //ShallowArray<ShallowArray< int*, Array::MemLocation::UNIFIED_MEM >, Array::MemLocation::UNIFIED_MEM> um_neighbors;
      int* um_value; //to be allocated using cudaMallocManaged
      int* um_publicValue; //to be allocated using cudaMallocManaged
      std::vector<int*> um_neighbors;
      ShallowArray_Flat<ShallowArray< int*, Array_Flat::MemLocation::UNIFIED_MEM >, Array_Flat::MemLocation::UNIFIED_MEM> um_neighbors;
      ShallowArray<LifeNode, 1000, 4> _nodes;
#else
      ShallowArray<LifeNode, 1000, 4> _nodes;
#endif
    */
   {
      MacroConditional gpuConditional(GPUCONDITIONAL);
      gpuConditional.setNegateCondition();
      CustomAttribute* nodes = new CustomAttribute("_nodes", "ShallowArray<" + getInstanceName() + ", 1000, 4>");
      nodes->setMacroConditional(gpuConditional);
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
   }
   {
      MacroConditional gpuConditional(GPUCONDITIONAL);
      CustomAttribute* nodes = new CustomAttribute("_nodes", "ShallowArray_Flat<" + getInstanceName() + ", Array_Flat<int>::MemLocation::CPU, 1000>");
      nodes->setMacroConditional(gpuConditional);
      // QUESTION: should this shallow array be setOwned? JK, RR, DL 11/29/05
      std::auto_ptr<Attribute> nodesAp(nodes);

      //std::ostringstream nodesDeleteString;
      //nodesDeleteString << "#ifdef HAVE_MPI\n"
      //   << TAB << "std::map<int, CCDemarshaller*>::iterator end2 = _demarshallerMap.end();\n"
      //   << TAB << "for (std::map<int, CCDemarshaller*>::iterator iter2=_demarshallerMap.begin(); iter2!=end2; ++iter2) {\n"
      //   << TAB << TAB << "delete (*iter2).second;\n"
      //   << TAB << "}\n"
      //   << "#endif\n"
      //   << TAB << "if (CG_sharedMembers) {\n"
      //   << TAB << TAB << "delete CG_sharedMembers;\n"
      //   << TAB << TAB << "CG_sharedMembers=0;\n"
      //   << TAB << "}\n";

      //nodes->setCustomDeleteString(nodesDeleteString.str());
      nodes->setAccessType(AccessType::PROTECTED);
      instance.addAttribute(nodesAp);
   }

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
