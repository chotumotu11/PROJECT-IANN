#Author: Drona Banerjee
#Name of Apllication: proOCR
#Name of python script bla.py
#Purpose:
#Tesseract is a command line engine and does not provide any sort of GUI. This limits the number of users of this OCR engine, 
#as it can be a daunting task to get it up  and running for someone with little knowledge about the OCR technology.
#This is an attempt to develop a moderately interactive GUI to make it more User-Friendly.


#NOTE: install Python 2.7 to run the python script(bla.py). The application WILL NOT work properly in higher versions of python(Python 3.X)

"""OCR with moderately interactive GUI in Python using the PyTesser Library from Google
http://code.google.com/p/pytesser/
by Drona Banerjee
V 0.0.1, 22/05/2017"""

"""
The PyTesser library (https://code.google.com/p/pytesser/downloads/detail?name=pytesser_v0.0.1.zip&can=2&q=), the tool that allows Python to work with Tesser

Tesseract is a command line engine and does not provide any sort of GUI. This limits the number of users of this OCR engine, 
as it can be a daunting task to get it up  and running for someone with little knowledge about the OCR technology.

This is an attempt to develop a moderately interactive GUI to make it more User-Friendly.

"""

from PIL import Image #pillow package for image processing
from pytesser import * # Library responsible for the magic
from Tkinter import Tk 
#from Tkinter import * # I have no idea why....but it can cause conflicts with Image module of Pillow package. Hence we import only Tk
from tkFileDialog import askopenfilename
import tkMessageBox #for message box
import Tkinter
import csv 
import easygui
import ttk

#Tk().withdraw() 
#filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
#print(filename)

filename= easygui.fileopenbox() # show an "Open" dialog box and return the name of the selected file
								# please refrain from picking invalid files. No error checking. Program wont crash. But you will 
								# probably get a blank file



# user is asked to provide a name of his/her choice to the output text file which contains the result.
the_name=easygui.enterbox("SPECIFY A NAME FOR OUTPUT FILE","OUTPUT FILE NAME")

image_file = filename

#using Image from Pillow to open the image file selected by user
im = Image.open(image_file)

# image_to_string() function of pytesser library is called to extract string from image file.
text = image_to_string(im) #for low dpi images
text = image_file_to_string(image_file) #for high dpi images

# to detect errors like "dpi less than 70. Not enough dpi"
text = image_file_to_string(image_file, graceful_errors=True)



# displays the output(the string after extraction from image file) in the command window
print "=====output=======\n"
print text


# creating a new file with file name specified by user and writing the string that has been generated in that file.
f= open(the_name,"w+")
f.write(text)
f.close() #closing the file.


print the_name
print "=====NOTE=======\n"
print "you will find the extracted text in the file named ",the_name," as well"



 



