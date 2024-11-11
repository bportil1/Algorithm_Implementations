#!/bin/bash

cppCheckScript (){
	for file in $(find . -name "*.cpp");
	do
		echo $file	
		cppcheck --enable=all $file 2> report.txt
	done
}

cppCheckScript
