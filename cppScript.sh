#!/bin/bash

cppCheckScript (){
	for file in $(find . -name "*.cpp");
	do
		echo $file	
		cppcheck --enable=all $file 2>> report.txt
	done
}

parseAnalysis(){
	file='report.txt'
	for line in $(grep ".*msrc/OpenEVSE.*" $file);
	do 
		echo $line
		echo $line >> parsed_report.txt
	done

}

#cppCheckScript

parseAnalysis
