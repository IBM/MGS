// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "MemberToInterface.h"
#include "InterfaceMapping.h"
#include "Interface.h"
#include "DataType.h"
#include "StructType.h"
#include "GeneralException.h"
#include "InternalException.h"
#include "NotFoundException.h"
#include "DuplicateException.h"
#include "MemberContainer.h"
#include "Class.h"
#include "Method.h"
#include "Constants.h"
#include <memory>
#include <string>
#include <map>
#include <iostream>
#include <sstream>

MemberToInterface::MemberToInterface(Interface* interface) 
   : InterfaceMapping(interface)
{
}

void MemberToInterface::duplicate(std::unique_ptr<MemberToInterface>&& rv) const
{
   rv.reset(new MemberToInterface(*this));
}

void MemberToInterface::duplicate(std::unique_ptr<InterfaceMapping>&& rv) const
{
   rv.reset(new MemberToInterface(*this));
}

bool MemberToInterface::checkAllMapped() 
{
   	bool retVal = true;
   	MemberContainer<DataType>::const_iterator it, 
   	   end = _interface->getMembers().end();
   	for (it = _interface->getMembers().begin(); it != end; it++) {
    	if (!existsInMappings(it->first)) {
	 		retVal = false;
	 		break;	 
      	}
   	}
   	return retVal;
}

void MemberToInterface::checkAndExtraWork(const std::string& name,
   	DataType* member, const DataType* interface, bool amp) {
   	const std::vector<std::string>& subAttributePath = 
    	member->getSubAttributePath();
   
   	if (subAttributePath.size() > 0) {
		StructType* nextStruct;
		DataType* nextMember;
		nextStruct = dynamic_cast<StructType*>(member);

		if (nextStruct == 0) {
			std::ostringstream os;
			os << member->getName()
				<< " is not a struct type"; 
			throw GeneralException(os.str());
		}

		if (nextStruct->isPointer()) {
			std::ostringstream os;
			os << nextStruct->getName()
				<< " can not be a pointer"; 
			throw GeneralException(os.str());
      	}
      
      	std::vector<std::string>::const_iterator it, next, end 
	 		= subAttributePath.end();
      	for (it = subAttributePath.begin(); it != end; ++it) { 
	 		try { 
			    nextMember = nextStruct->_members.getMember(*it);
	 		} 
			catch (NotFoundException& e) {
				std::ostringstream os;
				os << *it << " does not exist in struct " 
					<< nextStruct->getTypeName(); 
				throw GeneralException(os.str());
			}
		next = it + 1;
		if (next == end) {
			std::string memberTypeString = nextMember->getTypeString();
			if (amp) {
				memberTypeString += "*";
			}
			if (memberTypeString != interface->getTypeString()) {
				std::ostringstream os;
				os << " interface " << _interface->getName() << "'s member " 
					<<  name 
					<< " is of type " << interface->getDescriptor() << " not " 
					<< memberTypeString << " ( " 
					<< nextMember->getName() << "'s type)";
				throw GeneralException(os.str());
			}    
		}
		else {
			nextStruct = dynamic_cast<StructType*>(nextMember);
			if (nextStruct == 0) {
				std::ostringstream os;
				os << nextMember->getName()
					<< " is not a struct type"; 
				throw GeneralException(os.str());
			}	    

			if (nextStruct->isPointer()) {
				std::ostringstream os;
				os << nextStruct->getName()
					<< " can not be a pointer"; 
				throw GeneralException(os.str());
			}
		}
      }
   } 
   else {
		// The implemented interface dataType has to be the pointer of this
		// type, we'll serve &x.
		std::string memberTypeString = member->getTypeString();
		if (amp) {
			memberTypeString += "*";
		}
		std::string interfaceTypeString = interface->getTypeString();
		if (memberTypeString != interfaceTypeString) {
			std::ostringstream os;
			os << " interface " << _interface->getName() << "'s member " <<  name 
				<< " is of type " << interfaceTypeString << " not " 
				<< memberTypeString;
			throw GeneralException(os.str());
		}
   }
}

MemberToInterface::~MemberToInterface() 
{
}
 

std::string MemberToInterface::getMemberToInterfaceString(
   const std::string& interfaceName) const 
{
   return commonGenerateString(interfaceName, " << ", "\t");
}

void MemberToInterface::setupAccessorMethods(Class& instance) const
{
   const_iterator it, end = _mappings.end();
   if (instance.getClassInfoPrimeType() == Class::PrimeType::Node) {
    	for (it = _mappings.begin(); it != end; ++it) {
			const std::vector<std::string>& subAttributePath = 
				it->getDataType()->getSubAttributePath();
			std::string path = "";
			std::vector<std::string>::const_iterator sit, send 
				= subAttributePath.end();
			for (sit = subAttributePath.begin(); sit != send; ++sit) { 
				path += "." + *sit;
			}
			if (it->getDataType()->isArray()) {
				//create 2 methods [one used in HAVE_GPU and one is not]
				{
					//#if defined(HAVE_GPU)
					// convert ShallowArray< double >* 
					// into 
					// ShallowArray_Flat< double, Array_Flat<int>::MemLocation::UNIFIED_MEM>*
					std::string  type = it->getTypeString(); 
					std::string from = "ShallowArray<";
					std::string to = "ShallowArray_Flat<";
					type = type.replace(type.find(from),from.length(),to);
					std::size_t start = type.find_first_of("<");
					std::size_t last = type.find_first_of(">");
					std::string element_datatype = type.substr(start+1, last-start-1);
					type = type.replace(start+1, last-start-1, element_datatype + ", " + MEMORY_LOCATION);
					std::unique_ptr<Method> method(
						new Method(PREFIX + "get_" + _interface->getName() + "_" + it->getName(), type));
					MacroConditional gpuConditional(GPUCONDITIONAL);
					method->setMacroConditional(gpuConditional);
					std::string name = it->getDataType()->getName();
					if (it->getDataType()->isShared()) {
						name = "getNonConstSharedMembers()." + name;
					}
					name += path;
					if (it->getDataType()->isShared()) {
					method->setFunctionBody(
						(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
						+ TAB + "return " 
						+ (it->getNeedsAmpersand() ? "&" : "") 
						+ name + ";\n");
					}
					else {
						//TUAN: only accept non-shared data on GPU for now
						// maybe in the future we want to define shared data as '__constant__' or ...
						std::string body("");
						body = (!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
							+ TAB + "return " 
							+ (it->getNeedsAmpersand() ? "&" : "") 
							+ "("+REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME  
							+ name + "["+REF_INDEX+"]);\n";
						method->setFunctionBody(body);
					}
					method->setVirtual();
					instance.addMethod(std::move(method));
				}
				{
					//#if ! defined(HAVE_GPU)
					std::unique_ptr<Method> method(
						new Method(PREFIX + "get_" + _interface->getName() + "_" + it->getName(),
						it->getTypeString()));
					MacroConditional gpuConditional(GPUCONDITIONAL);
					gpuConditional.setNegateCondition();
					method->setMacroConditional(gpuConditional);
					std::string name = it->getDataType()->getName();
					if (it->getDataType()->isShared()) {
						name = "getNonConstSharedMembers()." + name;
					}
					name += path;
					if (it->getDataType()->isShared()) {
						method->setFunctionBody(
							(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
							+ TAB + "return " 
							+ (it->getNeedsAmpersand() ? "&" : "") 
							+ name + ";\n");
					}
					else {
						//TUAN: only accept non-shared data on GPU for now
						// maybe in the future we want to define shared data as '__constant__' or ...
						std::string body("");
						//if (instance.getClassInfoPrimeType() == Class::PrimeType::Node)
						//{
						//   body = STR_GPU_CHECK_START 
						//      +
						//      (!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
						//      + TAB + "return " 
						//      + (it->getNeedsAmpersand() ? "&" : "") 
						//      + "("+REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME  
						//      + name + "["+REF_INDEX+"]);\n"
						//      +
						//      "#else\n";
						//}
						body +=
							(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
							+ TAB + "return " 
							+ (it->getNeedsAmpersand() ? "&" : "") 
							+ name + ";\n";
						//if (instance.getClassInfoPrimeType() == Class::PrimeType::Node)
						//{
						//   body +=
						//      STR_GPU_CHECK_END;
						//}
						method->setFunctionBody(body);
					}
					method->setVirtual();
					instance.addMethod(std::move(method));
				}
			}
			else {
				std::unique_ptr<Method> method(
				new Method(PREFIX + "get_" + _interface->getName() + "_" + it->getName(),
					it->getTypeString()));
				std::string name = it->getDataType()->getName();
				if (it->getDataType()->isShared()) {
					name = "getNonConstSharedMembers()." + name;
				}
				name += path;
				if (it->getDataType()->isShared()) {
					method->setFunctionBody(
						(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
						+ TAB + "return " 
						+ (it->getNeedsAmpersand() ? "&" : "") 
						+ name + ";\n");
				}
				else {
					//TUAN: only accept non-shared data on GPU for now
					// maybe in the future we want to define shared data as '__constant__' or ...
					std::string body("");
					if (instance.getClassInfoPrimeType() == Class::PrimeType::Node) {
						body = STR_GPU_CHECK_START +
							(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
							+ TAB + "return " 
							+ (it->getNeedsAmpersand() ? "&" : "") 
							+ "("+REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME  
							+ name + "["+REF_INDEX+"]);\n"
							+ "#else\n";
					}
					body += (!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
						+ TAB + "return " 
						+ (it->getNeedsAmpersand() ? "&" : "") 
						+ name + ";\n";
					if (instance.getClassInfoPrimeType() == Class::PrimeType::Node) {
						body += STR_GPU_CHECK_END;
					}
					method->setFunctionBody(body);
				}
				method->setVirtual();
				instance.addMethod(std::move(method));
			}
		}      
 	}
   	else {
    	for (it = _mappings.begin(); it != end; ++it) {
			const std::vector<std::string>& subAttributePath = 
			it->getDataType()->getSubAttributePath();
			std::string path = "";
			std::vector<std::string>::const_iterator sit, send = subAttributePath.end();
			for (sit = subAttributePath.begin(); sit != send; ++sit) { 
				path += "." + *sit;
			}
			std::unique_ptr<Method> method(
				new Method(PREFIX + "get_" + _interface->getName() + "_" + it->getName(),
				it->getTypeString()));
			std::string name = it->getDataType()->getName();
			if (it->getDataType()->isShared()) {
				name = "getNonConstSharedMembers()." + name;
			}
			name += path;
			if (it->getDataType()->isShared()) {
				method->setFunctionBody(
				(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
				+ TAB + "return " 
				+ (it->getNeedsAmpersand() ? "&" : "") 
				+ name + ";\n");
			}
			else {
				//TUAN: only accept non-shared data on GPU for now
				// maybe in the future we want to define shared data as '__constant__' or ...
				std::string body("");
				if (instance.getClassInfoPrimeType() == Class::PrimeType::Node) {
					body = STR_GPU_CHECK_START 
					+ (!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
					+ TAB + "return " 
					+ (it->getNeedsAmpersand() ? "&" : "") 
					+ "("+REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME  
					+ name + "["+REF_INDEX+"]);\n"
					+ "#else\n";
				}
				else if (instance.getClassInfoPrimeType() == Class::PrimeType::Constant) {
					//Constant
					std::string attName = PREFIX_MEMBERNAME + "constant_"+ instance.getName().substr(3)  + "_" + name;
					body = STR_GPU_CHECK_START +
					(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
					+ TAB + "return " 
					+ (it->getNeedsAmpersand() ? "&" : "") 
					+ "(" + attName + "["+REF_INDEX+"]);\n"
					+ "#else\n";
				}
				body +=
					(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
					+ TAB + "return " 
					+ (it->getNeedsAmpersand() ? "&" : "") 
					+ name + ";\n";
				if (instance.getClassInfoPrimeType() == Class::PrimeType::Node
					or instance.getClassInfoPrimeType() == Class::PrimeType::Constant) {
					body +=
					STR_GPU_CHECK_END;
				}
				method->setFunctionBody(body);
			}
			method->setVirtual();
			instance.addMethod(std::move(method));
		}      
   	}
}

void MemberToInterface::setupProxyAccessorMethods(Class& instance) const
{
	const_iterator it, end = _mappings.end();
   	for (it = _mappings.begin(); it != end; ++it) {
    	const std::vector<std::string>& subAttributePath = 
		it->getDataType()->getSubAttributePath();
    	std::string path = "";
	    std::vector<std::string>::const_iterator sit, send = subAttributePath.end();
	    for (sit = subAttributePath.begin(); sit != send; ++sit) { 
	 		path += "." + *sit;
	    }
    	if (it->getDataType()->isArray()) {
			{
				//#if defined(HAVE_GPU)
				// convert ShallowArray< double >* 
				// into 
				// ShallowArray_Flat< double, Array_Flat<int>::MemLocation::UNIFIED_MEM>*
				std::string  type = it->getTypeString(); 
				std::string from = "ShallowArray<";
				std::string to = "ShallowArray_Flat<";
				type = type.replace(type.find(from),from.length(),to);
				std::size_t start = type.find_first_of("<");
				std::size_t last = type.find_first_of(">");
				std::string element_datatype = type.substr(start+1, last-start-1);
				type = type.replace(start+1, last-start-1, element_datatype + ", " + MEMORY_LOCATION);
				std::unique_ptr<Method> method(
				new Method(PREFIX + "get_" + _interface->getName() + "_" + it->getName(), type));
				MacroConditional gpuConditional(GPUCONDITIONAL);
				method->setMacroConditional(gpuConditional);
				std::string name = it->getDataType()->getName();
				if (it->getDataType()->isShared()) {
					name = "getNonConstSharedMembers()." + name;
				}
				name += path;
				if (instance.getClassInfoPrimeType() != Class::PrimeType::Node) {
					method->setFunctionBody(
						TAB + "return &" 
						+ name + ";\n");
				}
			    else {
					if (it->getDataType()->isShared()) {
					/* (Oct-20-2019) IMPORTANT: only support exporting a pointer via SHARED data only */
					method->setFunctionBody(
						(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
						+ TAB + "return " 
						+ (it->getNeedsAmpersand() ? "&" : "") 
						+ name + ";\n");
					}
					else {
						//TUAN: proxy only have non-shared data on GPU for now
						// maybe in the future we want to define shared data as '__constant__' or ...
						std::string body;
						body = "#if PROXY_ALLOCATION == OPTION_3\n"
							+ TAB + "return &" 
							+ "(("+REF_CC_OBJECT+"->" + GETDEMARSHALLER_FUNC_NAME + "(" 
							+ REF_DEMARSHALLER_INDEX + "))->" + PREFIX_MEMBERNAME  
							+ name + "["+REF_INDEX+"]);\n"
							+ "#elif PROXY_ALLOCATION == OPTION_4\n"
							+ TAB + "return &" 
							+ "("+REF_CC_OBJECT+"->" + PREFIX_PROXY_MEMBERNAME
							+ name + "["+REF_INDEX+"]);\n"
							+ "#endif\n\n";
						method->setFunctionBody(body);
				    }
	 			}
				method->setVirtual();
				instance.addMethod(std::move(method));
			}
			{
				//#if ! defined(HAVE_GPU)
				std::unique_ptr<Method> method(
				new Method(PREFIX + "get_" + _interface->getName() + "_" + it->getName(),
					it->getTypeString()));
				MacroConditional gpuConditional(GPUCONDITIONAL);
				gpuConditional.setNegateCondition();
				method->setMacroConditional(gpuConditional);
				std::string name = it->getDataType()->getName();
				if (it->getDataType()->isShared()) {
					name = "getNonConstSharedMembers()." + name;
				}
				name += path;
				if (instance.getClassInfoPrimeType() != Class::PrimeType::Node) {
					method->setFunctionBody(
						TAB + "return &" 
						+ name + ";\n");
				}
				else {
					if (it->getDataType()->isShared()) {
					/* IMPORTANT: only support exporting a pointer via SHARED data only */
					method->setFunctionBody(
						(!it->getNeedsAmpersand() ? TAB + "assert(" + name + ");\n" : "")
						+ TAB + "return " 
						+ (it->getNeedsAmpersand() ? "&" : "") 
						+ name + ";\n");
					}
					else {
						std::string body;
						body = "return &" + name + ";\n";
						method->setFunctionBody(body);
					}
				}
				method->setVirtual();
				instance.addMethod(std::move(method));
			}
      	}
	  	else {
			std::unique_ptr<Method> method(
				new Method(PREFIX + "get_" + _interface->getName() + "_" + it->getName(),
				it->getTypeString()));
			std::string name = it->getDataType()->getName();
			if (it->getDataType()->isShared()) {
				name = "getNonConstSharedMembers()." + name;
			}
			name += path;
			if (instance.getClassInfoPrimeType() != Class::PrimeType::Node) {
				method->setFunctionBody(
				TAB + "return &" 
				+ name + ";\n");
			}
			else {
				if (it->getDataType()->isShared()) {
				method->setFunctionBody(
					TAB + "return &" 
					+ name + ";\n");
				}
				else {
					//TUAN: proxy only have non-shared data on GPU for now
					// maybe in the future we want to define shared data as '__constant__' or ...
					std::string body;
					body = STR_GPU_CHECK_START 
						+ "#if PROXY_ALLOCATION == OPTION_3\n"
						+ TAB + "return &" 
						+ "(("+REF_CC_OBJECT+"->" + GETDEMARSHALLER_FUNC_NAME + "(" 
						+ REF_DEMARSHALLER_INDEX + "))->" + PREFIX_MEMBERNAME  
						+ name + "["+REF_INDEX+"]);\n"
						+ "#elif PROXY_ALLOCATION == OPTION_4\n"
						+ TAB + "return &" 
						+ "("+REF_CC_OBJECT+"->" + PREFIX_PROXY_MEMBERNAME
						+ name + "["+REF_INDEX+"]);\n"
						+ "#endif\n\n"
						+ "#else\n"
						+ TAB + "return &" 
						+ name + ";\n"
						+ STR_GPU_CHECK_END;
					method->setFunctionBody(body);
				}
			}
			method->setVirtual();
			instance.addMethod(std::move(method));
		}

   	}
}

bool MemberToInterface::hasMemberDataType(const std::string& name) const
{
   	std::vector<InterfaceMappingElement>::const_iterator it, 
    	end = _mappings.end();
   	for (it = _mappings.begin(); it != end; ++it) {
		const DataType* dt = it->getDataType();
    	if (!dt->isShared()) {
			if (dt->getName() == name) {
				return true;
	 		}
      	}
   	}
   	return false;
}

std::string MemberToInterface::getServiceNameCode(
   const std::string& tab) const
{
	std::ostringstream os;
   	os << tab << "if (" << INTERFACENAME << " == \"" 
    	<< _interface->getName() << "\") {\n";
   	std::string newTab = tab + TAB;
   	std::vector<InterfaceMappingElement>::const_iterator it, 
    	end = _mappings.end();
   	for (it = _mappings.begin(); it != end; ++it) {
    	os << it->getServiceNameCode(newTab);
	}
   	os << tab << "}\n";
   	return os.str();
}
