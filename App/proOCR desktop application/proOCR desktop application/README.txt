How to run proOCR desktop application.

The scripts in this directory make it possible to extract text from an image file(.jpg,.png,tif) and generate a searchable and editable text file 
with the extracted text in it, using a moderately interactive GUI.

Prerequisites:

** install Python 2.7 to run the python script(bla.py). The application may not work properly in higher versions of python(Python 3.X)
** install Tkinter, a package that has pyhton GUI libraries.
** install Tesseract 3.05.00
** install PIL, a package that allows to deal with images.
   instructions to install a package can be found in the link
   (http://fosshelp.blogspot.com.es/2013/04/how-to-convert-jpg-to-tiff-for-ocr-with.html)
** download the PyTesser library which enables the python script to interact with the tesseract engine(PyTesser may already exist in some Anaconda distributions)
   it can be downloaded from https://code.google.com/p/pytesser/downloads/detail?name=pytesser_v0.0.1.zip&can=2&q=
   The PyTesser library (https://code.google.com/p/pytesser/downloads/detail?name=pytesser_v0.0.1.zip&can=2&q=), the tool that allows Python to work with Tesser

Build instructions:


** Unzip the file(The PyTesser library (https://code.google.com/p/pytesser/downloads/detail?name=pytesser_v0.0.1.zip&can=2&q=),) on the same dir where 
   you have your python script(bla.py)
** place the image file from which you want to extract text in the same dir where 
   you have your python script(bla.py)
   
Run instructions:

** (if you do not have an IDE setup in your system)open command window in the directory where python 2.7 is installed
** type the command: python bla.py
** this will open up a file browser from which user can pick files to extract text from.
** once a file is chosen, user is asked to provide a name of his/her choice to the output text file which contains the result.
** the output result is also shown in the command window, along with the specified file name of the output text file which contains the result.




Other Dependencies and Licenses:
================================

proOCR uses Leptonica library (http://leptonica.com/) which essentially
uses a BSD 2-clause license. (http://leptonica.com/about-the-license.html)


All the code in this distribution is now licensed under the Apache License:

** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
** http://www.apache.org/licenses/LICENSE-2.0
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.




